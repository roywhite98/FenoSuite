# -*- coding: utf-8 -*-
"""数据集规范：dataset.json / samples.json 的数据结构。

目录布局（相对路径均以数据集目录为基准，使用 ``/`` 分隔）::

    <datasets_root>/<dataset_id>/
        dataset.json   名称、分割类型、类别定义、格式版本、来源记录
        samples.json   原图与掩码的对应关系、样本状态
        images/        原始图像（未划分的样本）
        masks/         标注掩码（语义：类别编号图；实例：masks/<样本>/inst_XXX.png 二值图）
        train/ val/ test/
            images/    已划分到该子集的原图
            masks/     对应掩码，结构同上
        reports/       导入、转换、校验报告

未划分与已划分的样本可以并存；样本主名在整个数据集内唯一，移动划分时文件名不变。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

FORMAT_VERSION = 1
DATASET_FILE = "dataset.json"
SAMPLES_FILE = "samples.json"
IMAGES_DIR = "images"
MASKS_DIR = "masks"
REPORTS_DIR = "reports"

TaskType = Literal["semantic", "instance"]
SampleStatus = Literal["unlabeled", "annotated", "predicted"]

TASK_TYPES: tuple[str, ...] = ("semantic", "instance")
TASK_TYPE_LABELS = {"semantic": "语义分割", "instance": "实例分割"}
SAMPLE_STATUSES: tuple[str, ...] = ("unlabeled", "annotated", "predicted")
STATUS_LABELS = {"unlabeled": "未标注", "annotated": "已标注", "predicted": "预测/候选"}

SPLITS: tuple[str, ...] = ("train", "val", "test")
SPLIT_LABELS = {"train": "训练集", "val": "验证集", "test": "测试集"}
UNSPLIT_LABEL = "未划分"

BACKGROUND_ID = 0


def normalize_split(split: str | None) -> str | None:
    """空值表示未划分；其它值必须是 train / val / test 之一。"""
    if split is None or split == "":
        return None
    s = str(split).strip().lower()
    if s not in SPLITS:
        raise ValueError(f"未知数据划分: {split}（应为 {', '.join(SPLITS)}）")
    return s


def split_label(split: str | None) -> str:
    return SPLIT_LABELS.get(split, UNSPLIT_LABEL) if split else UNSPLIT_LABEL

DEFAULT_PALETTE: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),
    (0, 200, 0),
    (230, 60, 60),
    (40, 120, 255),
    (255, 190, 0),
    (180, 60, 220),
    (0, 200, 200),
    (255, 120, 180),
    (140, 90, 40),
    (120, 200, 80),
)


def default_color(class_id: int) -> tuple[int, int, int]:
    if 0 <= class_id < len(DEFAULT_PALETTE):
        return DEFAULT_PALETTE[class_id]
    h = (class_id * 2654435761) & 0xFFFFFF
    return ((h >> 16) & 0xFF, (h >> 8) & 0xFF, h & 0xFF)


@dataclass
class LabelClass:
    """类别定义。``id`` 即语义掩码中存储的像素值；``color`` 只用于界面显示与彩色导出。"""

    id: int
    name: str
    color: tuple[int, int, int] = (0, 200, 0)

    def to_dict(self) -> dict[str, Any]:
        return {"id": int(self.id), "name": self.name, "color": [int(c) for c in self.color]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LabelClass:
        cid = int(d["id"])
        color = d.get("color") or default_color(cid)
        return cls(id=cid, name=str(d.get("name", f"类别{cid}")), color=tuple(int(c) for c in color[:3]))


@dataclass
class InstanceRecord:
    """实例分割中的单个对象：一张二值掩码 + 类别。"""

    id: int
    class_id: int
    mask: str

    def to_dict(self) -> dict[str, Any]:
        return {"id": int(self.id), "class_id": int(self.class_id), "mask": self.mask}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> InstanceRecord:
        return cls(id=int(d["id"]), class_id=int(d["class_id"]), mask=str(d["mask"]))


@dataclass
class Sample:
    """一条明确的原图—掩码配对记录。"""

    id: str
    image: str
    mask: str | None = None
    status: str = "unlabeled"
    width: int | None = None
    height: int | None = None
    source: str | None = None
    instances: list[InstanceRecord] = field(default_factory=list)
    split: str | None = None

    @property
    def has_annotation(self) -> bool:
        return bool(self.mask) or bool(self.instances)

    @property
    def image_name(self) -> str:
        return self.image.rsplit("/", 1)[-1]

    @property
    def stem(self) -> str:
        name = self.image_name
        return name.rsplit(".", 1)[0] if "." in name else name

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "image": self.image, "mask": self.mask, "status": self.status}
        if self.width is not None and self.height is not None:
            d["width"] = int(self.width)
            d["height"] = int(self.height)
        if self.source:
            d["source"] = self.source
        if self.split:
            d["split"] = self.split
        if self.instances:
            d["instances"] = [i.to_dict() for i in self.instances]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Sample:
        return cls(
            id=str(d["id"]),
            image=str(d["image"]),
            mask=d.get("mask") or None,
            status=str(d.get("status", "unlabeled")),
            width=d.get("width"),
            height=d.get("height"),
            source=d.get("source"),
            instances=[InstanceRecord.from_dict(x) for x in d.get("instances", [])],
            split=normalize_split(d.get("split")),
        )


@dataclass
class DatasetMeta:
    id: str
    name: str
    task_type: str = "semantic"
    classes: list[LabelClass] = field(default_factory=list)
    description: str = ""
    created_at: str = ""
    updated_at: str = ""
    value_mapping: dict[str, int] = field(default_factory=dict)
    provenance: dict[str, Any] | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    format_version: int = FORMAT_VERSION

    def class_by_id(self, class_id: int) -> LabelClass | None:
        return next((c for c in self.classes if c.id == class_id), None)

    @property
    def class_ids(self) -> set[int]:
        return {c.id for c in self.classes}

    @property
    def foreground_classes(self) -> list[LabelClass]:
        return [c for c in self.classes if c.id != BACKGROUND_ID]

    def next_class_id(self) -> int:
        return max((c.id for c in self.classes), default=0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "id": self.id,
            "name": self.name,
            "task_type": self.task_type,
            "classes": [c.to_dict() for c in sorted(self.classes, key=lambda c: c.id)],
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "value_mapping": dict(self.value_mapping),
            "provenance": self.provenance,
            "history": list(self.history),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DatasetMeta:
        version = int(d.get("format_version", FORMAT_VERSION))
        if version > FORMAT_VERSION:
            raise ValueError(f"数据集格式版本 {version} 高于软件支持的版本 {FORMAT_VERSION}，请升级软件")
        task_type = str(d.get("task_type", "semantic"))
        if task_type not in TASK_TYPES:
            raise ValueError(f"未知分割类型: {task_type}")
        return cls(
            id=str(d["id"]),
            name=str(d.get("name", d["id"])),
            task_type=task_type,
            classes=[LabelClass.from_dict(c) for c in d.get("classes", [])],
            description=str(d.get("description", "")),
            created_at=str(d.get("created_at", "")),
            updated_at=str(d.get("updated_at", "")),
            value_mapping={str(k): int(v) for k, v in (d.get("value_mapping") or {}).items()},
            provenance=d.get("provenance"),
            history=list(d.get("history", [])),
            format_version=version,
        )


def default_classes() -> list[LabelClass]:
    return [LabelClass(0, "背景", default_color(0)), LabelClass(1, "根", default_color(1))]
