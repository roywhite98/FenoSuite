# -*- coding: utf-8 -*-
"""远端训练服务：接收任务、执行模型训练并提供状态与产物下载。"""
from __future__ import annotations

import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Dict, Any, Optional

from flask import Flask, request, jsonify, send_file

# 项目根（server 的上级）
ROOT = Path(__file__).resolve().parents[1]

app = Flask(__name__)
# 任务状态: job_id -> { status, log_lines, artifacts_dir, error?, config_path?, data_zip_path? }
_jobs: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()


def _run_training(job_id: str, config_yaml: str, data_path: Optional[str], data_zip_path: Optional[str], save_subdir: str):
    """在后台线程中执行训练。"""
    job = _jobs.get(job_id)
    if not job or job["status"] != "pending":
        return
    with _lock:
        job["status"] = "running"
    work_dir = None
    try:
        work_dir = Path(tempfile.mkdtemp(prefix="plant_phenotype_job_"))
        config_path = work_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(config_yaml)
        data_root = work_dir / "data"
        data_root.mkdir(parents=True, exist_ok=True)
        if data_zip_path and Path(data_zip_path).exists():
            import zipfile
            with zipfile.ZipFile(data_zip_path, "r") as z:
                z.extractall(data_root)
            # 常见结构: data/images, data/masks 或 根目录即 images/masks
            if (data_root / "images").is_dir():
                pass
            else:
                for d in data_root.iterdir():
                    if d.is_dir() and (d / "images").is_dir():
                        shutil.move(str(d / "images"), str(data_root / "images"))
                        if (d / "masks").is_dir():
                            shutil.move(str(d / "masks"), str(data_root / "masks"))
                        break
        elif data_path and Path(data_path).is_dir():
            # 服务器本地路径
            data_root = Path(data_path)
        else:
            raise ValueError("需要 data_path 或 data_zip 上传")

        save_dir = work_dir / "output"
        save_dir.mkdir(parents=True, exist_ok=True)
        # 覆盖 config 中的 data.root_dir 和 train.save_dir
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg.setdefault("data", {})["root_dir"] = str(data_root)
        cfg.setdefault("train", {})["save_dir"] = str(save_dir)
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

        log_lines = job.get("log_lines", [])
        from plant_phenotype.training.facade import get_facade
        try:
            get_facade().run_local(
                config_path=str(config_path),
                data_root=str(data_root),
                save_dir=str(save_dir),
                line_callback=lambda line: log_lines.append(line),
            )
            with _lock:
                job["log_lines"] = log_lines
                job["artifacts_dir"] = str(save_dir)
                job["status"] = "success"
        except Exception as run_err:
            log_lines.append(f"异常: {run_err}")
            with _lock:
                job["log_lines"] = log_lines
                job["artifacts_dir"] = str(save_dir)
                job["status"] = "failed"
                job["error"] = str(run_err)
    except Exception as e:
        with _lock:
            job["status"] = "failed"
            job["error"] = str(e)
            job["log_lines"] = job.get("log_lines", []) + [f"异常: {e}"]
    finally:
        if work_dir and work_dir.exists():
            pass  # 保留 work_dir 供下载，可后续由清理任务删除


@app.route("/")
def index():
    """根路径：用于浏览器或脚本快速确认服务已启动。"""
    return jsonify({
        "service": "plant_phenotype 远程训练服务",
        "docs": "POST /job 提交任务, GET /job/<job_id> 查询状态与日志",
    })


@app.route("/job", methods=["POST"])
def submit_job():
    """提交训练任务。Body: config (YAML 字符串或 config 文件), data_path (可选), 或 data_zip 上传。"""
    config_yaml = None
    if "config" in request.files and request.files["config"].filename:
        config_yaml = request.files["config"].read().decode("utf-8", errors="replace")
    if not config_yaml:
        config_yaml = request.form.get("config")
    if not config_yaml and request.is_json:
        config_yaml = request.json.get("config")
    if not config_yaml:
        return jsonify({"error": "缺少 config（请以 config 文件或 form 字段 config 提交）"}), 400
    data_path = request.form.get("data_path") or (request.json.get("data_path") if request.is_json else None)
    data_zip_path = None
    if "data_zip" in request.files:
        f = request.files["data_zip"]
        if f.filename:
            fd, path = tempfile.mkstemp(suffix=".zip", prefix="job_data_")
            f.save(path)
            data_zip_path = path
    job_id = str(uuid.uuid4())
    with _lock:
        _jobs[job_id] = {
            "status": "pending",
            "log_lines": [],
            "artifacts_dir": None,
            "error": None,
        }
    save_subdir = f"job_{job_id}"
    t = threading.Thread(
        target=_run_training,
        args=(job_id, config_yaml, data_path, data_zip_path, save_subdir),
    )
    t.daemon = True
    t.start()
    return jsonify({"job_id": job_id})


@app.route("/job/<job_id>", methods=["GET"])
def get_job(job_id):
    """查询任务状态与日志末尾。"""
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404
    lines = job.get("log_lines", [])
    return jsonify({
        "status": job["status"],
        "log_tail": lines[-100:] if len(lines) > 100 else lines,
        "error": job.get("error"),
    })


@app.route("/job/<job_id>/artifacts", methods=["GET"])
def list_artifacts(job_id):
    """列出产物文件。"""
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404
    adir = job.get("artifacts_dir")
    if not adir or not Path(adir).is_dir():
        return jsonify({"files": []})
    files = [f.name for f in Path(adir).iterdir() if f.is_file()]
    return jsonify({"files": files})


@app.route("/job/<job_id>/download/<filename>", methods=["GET"])
def download_artifact(job_id, filename):
    """下载指定产物文件。"""
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404
    adir = job.get("artifacts_dir")
    if not adir:
        return jsonify({"error": "no artifacts"}), 404
    path = Path(adir) / filename
    if not path.is_file() or path.parent.resolve() != Path(adir).resolve():
        return jsonify({"error": "file not found"}), 404
    return send_file(str(path), as_attachment=True, download_name=filename)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
