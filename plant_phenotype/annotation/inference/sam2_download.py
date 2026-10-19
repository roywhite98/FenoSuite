# -*- coding: utf-8 -*-
"""根据配置将 SAM2 ONNX 权重下载到本地缓存目录。"""
from __future__ import annotations

import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import yaml

# 与 sam2_download 同包层级上溯到 plant_phenotype 根目录
_PKG_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _PKG_ROOT / "resources" / "configs" / "sam2_weights.yaml"


def default_cache_dir() -> Path:
    base = os.environ.get("PLANT_PHENO_SAM2_CACHE", "").strip()
    if base:
        return Path(base).expanduser().resolve()
    return (Path.home() / ".cache" / "plant_phenotype" / "sam2_onnx").resolve()


def load_sam2_weight_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    path = config_path if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"SAM2 配置文件不存在: {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("sam2_weights.yaml 格式错误：应为 YAML 映射")
    for key in ("encoder_url", "decoder_url", "encoder_filename", "decoder_filename"):
        if key not in data or not str(data[key]).strip():
            raise ValueError(f"sam2_weights.yaml 缺少或为空: {key}")
    return data


def _download_url(
    url: str,
    dest: Path,
    chunk: int = 4 * 1024 * 1024,
    pulse: Optional[Callable[[], None]] = None,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "PlantPhenotype-SAM2/1.0"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            with open(part, "wb") as out:
                while True:
                    block = resp.read(chunk)
                    if not block:
                        break
                    out.write(block)
                    if pulse:
                        pulse()
        part.replace(dest)
    except Exception:
        if part.is_file():
            part.unlink(missing_ok=True)
        raise


def ensure_sam2_weights(
    *,
    config_path: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    status: Optional[Callable[[str], None]] = None,
    pulse: Optional[Callable[[], None]] = None,
) -> Tuple[Path, Path]:
    """
    若缓存中尚无文件则从配置 URL 下载。
    返回 (encoder_path, decoder_path)。
    """
    cfg = load_sam2_weight_config(config_path)
    cdir = (cache_dir or default_cache_dir()).resolve()
    cdir.mkdir(parents=True, exist_ok=True)

    enc_name = str(cfg["encoder_filename"]).strip()
    dec_name = str(cfg["decoder_filename"]).strip()
    enc_path = cdir / enc_name
    dec_path = cdir / dec_name

    def log(msg: str) -> None:
        if status:
            status(msg)

    if not enc_path.is_file() or enc_path.stat().st_size == 0:
        log("正在下载 SAM2 Encoder，请稍候…")
        try:
            _download_url(str(cfg["encoder_url"]).strip(), enc_path, pulse=pulse)
        except urllib.error.URLError as e:
            raise RuntimeError(f"下载 encoder 失败: {e}") from e
        except Exception as e:
            raise RuntimeError(f"下载 encoder 失败: {e}") from e

    if not dec_path.is_file() or dec_path.stat().st_size == 0:
        log("正在下载 SAM2 Decoder…")
        try:
            _download_url(str(cfg["decoder_url"]).strip(), dec_path, pulse=pulse)
        except urllib.error.URLError as e:
            raise RuntimeError(f"下载 decoder 失败: {e}") from e
        except Exception as e:
            raise RuntimeError(f"下载 decoder 失败: {e}") from e

    log("SAM2 权重已就绪")
    return enc_path, dec_path
