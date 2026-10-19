# -*- coding: utf-8 -*-
"""远端训练最小代码包：按导入闭包收集训练所需的源码，无需在服务器上部署整个项目。

命令行导出（例如供 HTTP 训练服务手动部署）::

    python -m plant_phenotype.training.bundle -o fenosuite_train.zip
    python -m plant_phenotype.training.bundle --server -o fenosuite_server.zip
"""
from __future__ import annotations

import argparse
import ast
import zipfile
from pathlib import Path
from typing import Iterable

PACKAGE = "plant_phenotype"
REPO_ROOT = Path(__file__).resolve().parents[2]

TRAIN_ENTRY_MODULES = (
    "plant_phenotype.training.train",
    "plant_phenotype.training.visualize",
)
SERVER_ENTRY_MODULES = TRAIN_ENTRY_MODULES + ("plant_phenotype.server.app",)


def _module_file(module: str) -> Path | None:
    base = REPO_ROOT.joinpath(*module.split("."))
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    return None


def _imported_modules(module: str, path: Path) -> Iterable[str]:
    """源码中出现的本包导入（含函数内的延迟导入）；`from x import y` 时 y 可能是子模块，一并返回。"""
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                base = ".".join(parts[: len(parts) - node.level + 1])
                full = f"{base}.{node.module}" if node.module else base
            else:
                full = node.module or ""
            yield full
            for alias in node.names:
                yield f"{full}.{alias.name}"


def collect_module_files(entry_modules: Iterable[str] = TRAIN_ENTRY_MODULES) -> list[Path]:
    """入口模块及其传递依赖（仅本包）对应的源码文件，含各级 `__init__.py`。"""
    seen: dict[str, Path] = {}
    stack = list(entry_modules)
    while stack:
        module = stack.pop()
        if module in seen or not (module == PACKAGE or module.startswith(PACKAGE + ".")):
            continue
        path = _module_file(module)
        if path is None:
            continue
        seen[module] = path
        parts = module.split(".")
        stack.extend(".".join(parts[:i]) for i in range(1, len(parts)))
        stack.extend(_imported_modules(module, path))
    return sorted(set(seen.values()))


def build_bundle_zip(
    dest: str | Path,
    entry_modules: Iterable[str] = TRAIN_ENTRY_MODULES,
    extra_files: Iterable[Path] = (),
) -> Path:
    """把最小代码包写入 zip，包内保持 `plant_phenotype/...` 相对结构。"""
    dest = Path(dest)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for f in [*collect_module_files(entry_modules), *extra_files]:
            z.write(f, f.relative_to(REPO_ROOT).as_posix())
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="导出远端训练所需的最小代码包")
    parser.add_argument("-o", "--output", default="fenosuite_train.zip")
    parser.add_argument("--server", action="store_true", help="同时包含 HTTP 训练服务（run_server.py）")
    args = parser.parse_args()
    if args.server:
        out = build_bundle_zip(args.output, SERVER_ENTRY_MODULES, [REPO_ROOT / "run_server.py"])
    else:
        out = build_bundle_zip(args.output)
    with zipfile.ZipFile(out) as z:
        n = len(z.namelist())
    print(f"已导出 {n} 个文件到 {out.resolve()}")


if __name__ == "__main__":
    main()
