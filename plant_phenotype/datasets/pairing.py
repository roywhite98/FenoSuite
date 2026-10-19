# -*- coding: utf-8 -*-
"""原图—掩码自动配对规则。

支持两种现有布局：
- ``same_dir``：原图与 ``<stem><后缀>.png`` 位于同一目录（标注工具旧约定）；
- ``split_dir``：``images/`` 与同级 ``masks/`` 分目录，掩码名为 ``<stem>.png`` 或 ``<stem><后缀>.png``，
  ``masks/`` 内允许再嵌套子目录。

两种布局都可以再按数据划分组织，如 ``train/images`` + ``train/masks``、``images/train`` + ``masks/train``
或 ``val/a.png`` + ``val/a_mask.png``；路径中的 train/val/test 目录名会被识别为样本的划分。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from plant_phenotype.image_io import DEFAULT_IMAGE_EXTENSIONS

SPLIT_ALIASES = {
    "train": "train", "training": "train",
    "val": "val", "valid": "val", "validation": "val",
    "test": "test", "testing": "test",
}

LAYOUTS = ("auto", "same_dir", "split_dir")
LAYOUT_LABELS = {
    "auto": "自动识别",
    "same_dir": "同目录（原图 + *_mask）",
    "split_dir": "images / masks 分目录",
}


@dataclass
class PairingRules:
    layout: str = "auto"
    mask_suffix: str = "_mask"
    images_subdir: str = "images"
    masks_subdir: str = "masks"
    recursive: bool = True

    def to_dict(self) -> dict:
        return {
            "layout": self.layout,
            "mask_suffix": self.mask_suffix,
            "images_subdir": self.images_subdir,
            "masks_subdir": self.masks_subdir,
            "recursive": self.recursive,
        }

    @classmethod
    def from_dict(cls, d: dict | None) -> PairingRules:
        d = d or {}
        return cls(
            layout=str(d.get("layout", "auto")),
            mask_suffix=str(d.get("mask_suffix", "_mask")),
            images_subdir=str(d.get("images_subdir", "images")),
            masks_subdir=str(d.get("masks_subdir", "masks")),
            recursive=bool(d.get("recursive", True)),
        )


@dataclass
class PairCandidate:
    image: Path
    mask: Path | None
    group: str = ""
    split: str | None = None


@dataclass
class PairingResult:
    layout: str
    pairs: list[PairCandidate] = field(default_factory=list)
    orphan_masks: list[Path] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)

    @property
    def n_paired(self) -> int:
        return sum(1 for p in self.pairs if p.mask is not None)

    @property
    def split_counts(self) -> dict[str, int]:
        """源目录中识别到的划分及原图数；键 ``""`` 为不在任何划分目录下的原图。没有划分时为空。"""
        counts = Counter(p.split or "" for p in self.pairs)
        return dict(counts) if any(k for k in counts) else {}

    @property
    def masks(self) -> list[Path]:
        return [p.mask for p in self.pairs if p.mask is not None]


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in DEFAULT_IMAGE_EXTENSIONS


def _iter_files(d: Path, recursive: bool):
    it = d.rglob("*") if recursive else d.iterdir()
    return sorted((p for p in it if _is_image(p)), key=lambda p: str(p).lower())


def _strip_suffix(stem: str, suffix: str) -> str:
    if suffix and stem.lower().endswith(suffix.lower()):
        return stem[: -len(suffix)]
    return stem


def _rank(p: Path, base: Path) -> tuple[int, int, str]:
    """同名掩码冲突时优先：浅层目录、无损格式。"""
    lossy = p.suffix.lower() in (".jpg", ".jpeg", ".webp", ".gif")
    return (len(p.relative_to(base).parts), int(lossy), str(p).lower())


def index_masks(mask_dir: Path, suffix: str, recursive: bool, conflicts: list[str] | None = None) -> dict[str, Path]:
    """以去掉后缀后的小写 stem 为键索引掩码目录。"""
    found: dict[str, list[Path]] = {}
    if not mask_dir.is_dir():
        return {}
    for p in _iter_files(mask_dir, recursive):
        found.setdefault(_strip_suffix(p.stem, suffix).lower(), []).append(p)
    out = {}
    for key, paths in found.items():
        paths.sort(key=lambda p: _rank(p, mask_dir))
        out[key] = paths[0]
        if len(paths) > 1 and conflicts is not None:
            others = ", ".join(str(x.relative_to(mask_dir)) for x in paths[1:])
            conflicts.append(f"掩码同名冲突 {key}: 使用 {paths[0].relative_to(mask_dir)}，忽略 {others}")
    return out


def _group_of(p: Path, source: Path, skip: str = "") -> str:
    try:
        rel = p.parent.relative_to(source)
    except ValueError:
        return ""
    return "_".join(x for x in rel.parts if not skip or x.lower() != skip.lower())


def split_of(p: Path, source: Path) -> str | None:
    """文件相对源目录的上级目录中，最靠近源目录的 train/val/test（及别名）目录名。"""
    try:
        parts = p.parent.relative_to(source).parts
    except ValueError:
        return None
    return next((SPLIT_ALIASES[x.lower()] for x in parts if x.lower() in SPLIT_ALIASES), None)


def _find_image_dirs(source: Path, rules: PairingRules) -> list[Path]:
    name = rules.images_subdir.strip().lower()
    if not name:
        return []
    dirs = []
    if source.name.lower() == name:
        dirs.append(source)
    if rules.recursive:
        candidates = list(source.rglob("*"))
    else:
        # 不递归时仍识别 <源目录>/<train|val|test>/images
        top = [d for d in source.iterdir() if d.is_dir()]
        candidates = top + [c for d in top if d.name.lower() in SPLIT_ALIASES for c in d.iterdir()]
    for d in candidates:
        if d.is_dir() and d.name.lower() == name:
            dirs.append(d)
    return sorted(set(dirs))


def detect_layout(source: Path, rules: PairingRules) -> str:
    return "split_dir" if _find_image_dirs(source, rules) else "same_dir"


def discover_pairs(source: str | Path, rules: PairingRules | None = None) -> PairingResult:
    source = Path(source)
    rules = rules or PairingRules()
    if not source.is_dir():
        raise FileNotFoundError(f"源目录不存在: {source}")
    layout = rules.layout if rules.layout != "auto" else detect_layout(source, rules)
    if layout == "split_dir":
        return _discover_split(source, rules)
    if layout == "same_dir":
        return _discover_same_dir(source, rules)
    raise ValueError(f"未知布局: {rules.layout}")


def _discover_same_dir(source: Path, rules: PairingRules) -> PairingResult:
    suffix = rules.mask_suffix
    if not suffix:
        raise ValueError("同目录布局需要掩码后缀（如 _mask）以区分原图与掩码")
    result = PairingResult(layout="same_dir")
    by_dir: dict[Path, list[Path]] = {}
    for p in _iter_files(source, rules.recursive):
        by_dir.setdefault(p.parent, []).append(p)
    for d, files in sorted(by_dir.items(), key=lambda kv: str(kv[0]).lower()):
        masks = [p for p in files if p.stem.lower().endswith(suffix.lower())]
        images = [p for p in files if p not in masks]
        mask_index: dict[str, list[Path]] = {}
        for m in masks:
            mask_index.setdefault(_strip_suffix(m.stem, suffix).lower(), []).append(m)
        used: set[Path] = set()
        for img in images:
            cands = sorted(mask_index.get(img.stem.lower(), []), key=lambda p: _rank(p, d))
            mask = cands[0] if cands else None
            if len(cands) > 1:
                result.conflicts.append(f"{img.name} 有多个掩码，使用 {mask.name}")
            if mask is not None:
                used.add(mask)
            result.pairs.append(PairCandidate(img, mask, _group_of(img, source), split_of(img, source)))
        result.orphan_masks.extend(m for m in masks if m not in used)
    return result


def _rel_parts(p: Path, base: Path) -> tuple[str, ...]:
    return tuple(x.lower() for x in p.parent.relative_to(base).parts)


def _discover_split(source: Path, rules: PairingRules) -> PairingResult:
    result = PairingResult(layout="split_dir")
    for img_dir in _find_image_dirs(source, rules):
        mask_dir = img_dir.parent / rules.masks_subdir
        found: dict[str, list[Path]] = {}
        if mask_dir.is_dir():
            for p in _iter_files(mask_dir, True):
                found.setdefault(_strip_suffix(p.stem, rules.mask_suffix).lower(), []).append(p)
        used: set[Path] = set()
        for img in _iter_files(img_dir, rules.recursive):
            img_sub, img_split = _rel_parts(img, img_dir), split_of(img, source)

            # 同名掩码优先取与原图相同子目录（如 images/train ↔ masks/train）、相同划分的那个
            def prefer(p: Path) -> tuple[bool, bool]:
                return _rel_parts(p, mask_dir) != img_sub, split_of(p, source) != img_split

            cands = sorted(found.get(img.stem.lower(), []), key=lambda p: (*prefer(p), *_rank(p, mask_dir)))
            mask = cands[0] if cands else None
            ties = [p for p in cands[1:] if prefer(p) == prefer(mask)]
            if ties:
                others = ", ".join(str(x.relative_to(mask_dir)) for x in ties)
                result.conflicts.append(
                    f"掩码同名冲突 {img.stem}: 使用 {mask.relative_to(mask_dir)}，忽略 {others}"
                )
            if mask is not None:
                if mask in used:
                    result.conflicts.append(f"{mask.name} 同时匹配多张原图，{img.name} 不配对")
                    mask = None
                else:
                    used.add(mask)
            result.pairs.append(PairCandidate(
                img, mask, _group_of(img, source, skip=rules.images_subdir), img_split,
            ))
        result.orphan_masks.extend(p for paths in found.values() for p in paths if p not in used)
    return result
