# -*- coding: utf-8 -*-
"""数据集管理：存储、配对、导入、校验、转换、保护规则与整体迁移。"""
from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from plant_phenotype.app.dataset_service import DatasetService  # noqa: E402
from plant_phenotype.datasets import convert, importer, mask_io, validation  # noqa: E402
from plant_phenotype.datasets.pairing import PairingRules, discover_pairs  # noqa: E402
from plant_phenotype.datasets.schema import LabelClass  # noqa: E402
from plant_phenotype.datasets.store import DatasetError, DatasetProtectedError, DatasetStore  # noqa: E402
from plant_phenotype.datasets.training_source import binary_training_pairs, binary_training_splits  # noqa: E402

try:
    import albumentations  # noqa: F401
    import torch  # noqa: F401

    _HAS_TRAINING = True
except ImportError:
    _HAS_TRAINING = False


def _img(path: Path, w: int = 16, h: int = 12, value: int = 100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((h, w, 3), value, dtype=np.uint8)).save(path)


def _gray_mask(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(path)


def _binary_arr(w: int = 16, h: int = 12) -> np.ndarray:
    a = np.zeros((h, w), dtype=np.uint8)
    a[2:6, 3:9] = 255
    return a


class DatasetTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.base = Path(self._td.name)
        self.root = self.base / "datasets"
        self.store = DatasetStore(self.root)

    def tearDown(self) -> None:
        self._td.cleanup()


class TestPairing(DatasetTestCase):
    def test_same_dir_layout(self) -> None:
        src = self.base / "src"
        _img(src / "a.png")
        _gray_mask(src / "a_mask.png", _binary_arr())
        _img(src / "b.jpg")
        _gray_mask(src / "orphan_mask.png", _binary_arr())
        res = discover_pairs(src, PairingRules())
        self.assertEqual(res.layout, "same_dir")
        pairs = {p.image.name: p.mask.name if p.mask else None for p in res.pairs}
        self.assertEqual(pairs, {"a.png": "a_mask.png", "b.jpg": None})
        self.assertEqual([p.name for p in res.orphan_masks], ["orphan_mask.png"])

    def test_split_layout_nested_masks_and_suffix(self) -> None:
        src = self.base / "src"
        _img(src / "images" / "x.png")
        _img(src / "images" / "y.png")
        _gray_mask(src / "masks" / "x_mask.png", _binary_arr())
        _gray_mask(src / "masks" / "deep" / "y.png", _binary_arr())
        res = discover_pairs(src, PairingRules())
        self.assertEqual(res.layout, "split_dir")
        self.assertEqual(res.n_paired, 2)


class TestStoreAndService(DatasetTestCase):
    def test_multiple_datasets_are_isolated(self) -> None:
        a = self.store.create("A")
        b = self.store.create("B")
        src = self.base / "img.png"
        _img(src)
        sa = a.add_image(src)
        a.save()
        self.assertNotEqual(a.path, b.path)
        self.assertEqual(len(self.store.load(a.id).samples), 1)
        self.assertEqual(len(self.store.load(b.id).samples), 0)
        a.save_binary_annotation(sa.id, _binary_arr() > 0)
        self.assertTrue(self.store.load(a.id).samples[0].has_annotation)
        self.assertFalse((b.masks_dir / f"{sa.stem}.png").exists())

        svc = DatasetService(self.root)
        svc.set_current(a.id)
        events: list = []
        svc.on_current_changed(events.append)
        svc.set_current(b.id)
        self.assertEqual(events, [b.id])
        self.assertEqual(DatasetService(self.root).current_id, b.id)

    def test_rename_keeps_id_and_files(self) -> None:
        ds = self.store.create("旧名")
        _img(self.base / "a.png")
        ds.add_image(self.base / "a.png")
        ds.save()
        self.store.rename(ds.id, "新名")
        again = self.store.load(ds.id)
        self.assertEqual(again.name, "新名")
        self.assertTrue(again.abs(again.samples[0].image).is_file())

    def test_copy_and_delete(self) -> None:
        ds = self.store.create("源")
        cp = self.store.copy(ds.id)
        self.assertNotEqual(cp.id, ds.id)
        self.assertEqual(cp.meta.provenance["source_dataset_id"], ds.id)
        self.store.delete(ds.id)
        self.assertEqual(self.store.list_ids(), [cp.id])

    def test_relocated_root_still_readable(self) -> None:
        src = self.base / "src"
        _img(src / "a.png")
        _gray_mask(src / "a_mask.png", _binary_arr())
        ds, _ = importer.import_dataset(
            self.store, src, importer.ImportOptions(name="迁移", mapping={"0": 0, "255": 1})
        )
        moved = self.base / "elsewhere" / "datasets"
        shutil.move(str(self.root), str(moved))
        again = DatasetStore(moved).load(ds.id)
        idx = again.load_index_mask(again.samples[0])
        self.assertEqual(int(idx.max()), 1)
        self.assertEqual(validation.validate_dataset(again), [])

    def test_legacy_dirs_detected(self) -> None:
        (self.root / "data" / "images").mkdir(parents=True)
        self.store.create("新")
        self.assertEqual([p.name for p in self.store.find_legacy_dirs()], ["data"])


class TestImportAndMapping(DatasetTestCase):
    def test_binary_import_becomes_index_mask(self) -> None:
        src = self.base / "src"
        _img(src / "images" / "a.png")
        _img(src / "images" / "b.png")
        _gray_mask(src / "masks" / "a.png", _binary_arr())
        res = discover_pairs(src)
        scan = importer.scan_mask_values(res.masks)
        self.assertEqual(scan.kind, "binary")
        mapping, classes = importer.suggest_mapping(scan)
        ds, report = importer.import_dataset(
            self.store, src, importer.ImportOptions(name="二值", mapping=mapping, classes=classes)
        )
        self.assertEqual(report["images"], 2)
        self.assertEqual(report["masks"], 1)
        self.assertTrue(Path(report["report_path"]).is_file())
        sample = next(s for s in ds.samples if s.has_annotation)
        self.assertEqual(sorted(np.unique(ds.load_index_mask(sample)).tolist()), [0, 1])
        self.assertEqual(sum(1 for s in ds.samples if s.status == "unlabeled"), 1)
        self.assertEqual(ds.meta.provenance["kind"], "import")

    def test_color_import_and_export_roundtrip_keeps_labels(self) -> None:
        src = self.base / "src"
        _img(src / "a.png")
        color = np.zeros((12, 16, 3), dtype=np.uint8)
        color[1:4, 1:4] = (255, 0, 0)
        color[6:10, 6:12] = (0, 0, 255)
        Image.fromarray(color).save(src / "a_mask.png")
        scan = importer.scan_mask_values(discover_pairs(src).masks)
        self.assertTrue(scan.is_color)
        mapping, classes = importer.suggest_mapping(scan)
        ds, report = importer.import_dataset(
            self.store, src, importer.ImportOptions(name="彩色", mapping=mapping, classes=classes)
        )
        self.assertEqual(report["failed"], [])
        self.assertEqual(len(ds.meta.foreground_classes), 2)
        out = self.base / "export"
        convert.export_dataset(ds, out, layout="same_dir", encoding="color")
        back = np.array(Image.open(out / "a_mask.png").convert("RGB"))
        np.testing.assert_array_equal(back, color)

    def test_unmapped_values_are_reported_not_silently_lost(self) -> None:
        src = self.base / "src"
        _img(src / "a.png")
        arr = _binary_arr()
        arr[0, 0] = 7
        _gray_mask(src / "a_mask.png", arr)
        ds, report = importer.import_dataset(
            self.store, src, importer.ImportOptions(name="缺映射", mapping={"0": 0, "255": 1})
        )
        self.assertEqual(report["masks"], 0)
        self.assertIn("7", report["failed"][0]["error"])
        self.assertFalse(ds.samples[0].has_annotation)


class TestValidation(DatasetTestCase):
    def test_detects_problems(self) -> None:
        ds = self.store.create("校验")
        _img(self.base / "a.png", 16, 12)
        _img(self.base / "b.png", 16, 12)
        a = ds.add_image(self.base / "a.png")
        b = ds.add_image(self.base / "b.png")
        ds.set_semantic_mask(a, np.zeros((10, 10), dtype=np.uint8))
        bad = np.zeros((12, 16), dtype=np.uint8)
        bad[0, 0] = 9
        ds.set_semantic_mask(b, bad)
        c = ds.add_image(self.base / "a.png")
        c.image = a.image
        ds.save()
        kinds = {i.kind for i in validation.validate_dataset(ds)}
        self.assertIn("size_mismatch", kinds)
        self.assertIn("unknown_label", kinds)
        self.assertIn("duplicate_image", kinds)
        self.assertIn("orphan_file", kinds)
        ds.abs(a.image).unlink()
        self.assertIn("missing_image", {i.kind for i in validation.validate_dataset(ds)})


class TestProtectionAndLabels(DatasetTestCase):
    def _multi(self):
        classes = [LabelClass(0, "背景"), LabelClass(1, "根"), LabelClass(2, "叶")]
        ds = self.store.create("多类", classes=classes)
        _img(self.base / "a.png")
        s = ds.add_image(self.base / "a.png")
        idx = np.zeros((12, 16), dtype=np.uint8)
        idx[1:3, 1:3] = 1
        idx[5:8, 5:8] = 2
        ds.set_semantic_mask(s, idx)
        ds.save()
        return ds, s, idx

    def test_binary_save_refuses_multiclass(self) -> None:
        ds, s, idx = self._multi()
        with self.assertRaises(DatasetProtectedError):
            ds.save_binary_annotation(s.id, np.ones((12, 16), bool))
        np.testing.assert_array_equal(ds.load_index_mask(s), idx)
        with self.assertRaises(DatasetProtectedError):
            binary_training_pairs(ds.path)

    def test_binary_save_refuses_to_overwrite_unknown_labels(self) -> None:
        ds = self.store.create("二值")
        _img(self.base / "a.png")
        s = ds.add_image(self.base / "a.png")
        idx = np.zeros((12, 16), dtype=np.uint8)
        idx[0, 0] = 5
        ds.set_semantic_mask(s, idx)
        with self.assertRaises(DatasetProtectedError):
            ds.save_binary_annotation(s.id, np.ones((12, 16), bool))

    def test_merge_preview_and_apply_then_training_allowed(self) -> None:
        ds, s, _ = self._multi()
        preview = ds.merge_preview([2], 1)
        self.assertEqual(preview["pixels"], 9)
        self.assertEqual(preview["samples_upper_bound"], 1)
        ds.merge_classes([2], 1)
        again = self.store.load(ds.id)
        self.assertEqual(sorted(np.unique(again.load_index_mask(again.samples[0])).tolist()), [0, 1])
        self.assertIsNone(again.protection_reason())
        self.assertEqual(len(binary_training_pairs(again.path)), 1)

    def test_delete_used_class_rejected(self) -> None:
        ds, _, _ = self._multi()
        with self.assertRaises(DatasetError):
            ds.delete_class(2)
        ds.rename_class(2, "叶片")
        ds.set_class_color(2, (1, 2, 3))
        self.assertEqual(ds.meta.class_by_id(2).color, (1, 2, 3))


class TestConversion(DatasetTestCase):
    def test_semantic_instance_roundtrip(self) -> None:
        classes = [LabelClass(0, "背景"), LabelClass(1, "根"), LabelClass(2, "叶")]
        src = self.store.create("语义", classes=classes)
        _img(self.base / "a.png")
        s = src.add_image(self.base / "a.png")
        idx = np.zeros((12, 16), dtype=np.uint8)
        idx[1:3, 1:3] = 1
        idx[1:3, 10:12] = 1
        idx[6:9, 5:8] = 2
        src.set_semantic_mask(s, idx)
        src.save()

        inst, rep = convert.semantic_to_instance(self.store, src.id, "实例", min_area=1)
        self.assertEqual(rep["instances"], 3)
        self.assertEqual(inst.samples[0].status, "predicted")
        self.assertIsNotNone(inst.protection_reason())
        sem, rep2 = convert.instance_to_semantic(self.store, inst.id, "回到语义")
        np.testing.assert_array_equal(sem.load_index_mask(sem.samples[0]), idx)
        self.assertEqual(sem.meta.provenance["source_dataset_id"], inst.id)
        np.testing.assert_array_equal(self.store.load(src.id).load_index_mask(s), idx)


class TestSamplesAndPredictions(DatasetTestCase):
    def test_same_stem_images_get_separate_masks(self) -> None:
        ds = self.store.create("同名")
        _img(self.base / "a.png")
        _img(self.base / "a.jpg")
        s1 = ds.add_image(self.base / "a.png")
        s2 = ds.add_image(self.base / "a.jpg")
        self.assertNotEqual(s1.stem, s2.stem)
        self.assertNotEqual(ds.mask_rel_for(s1), ds.mask_rel_for(s2))

    def _binary_with_samples(self):
        ds = self.store.create("预测")
        for n in ("a", "b", "c"):
            _img(self.base / "src" / f"{n}.png")
            ds.add_image(self.base / "src" / f"{n}.png")
        manual = np.zeros((12, 16), dtype=np.uint8)
        manual[0:2, 0:2] = 1
        ds.save_binary_annotation(ds.samples[0].id, manual)
        pred = self.base / "out" / "masks"
        for n in ("a", "b"):
            _gray_mask(pred / f"{n}_mask.png", _binary_arr())
        return ds, manual, pred

    def test_write_back_keeps_manual_annotations(self) -> None:
        ds, manual, pred = self._binary_with_samples()
        report = importer.write_back_predictions(ds, pred)
        self.assertEqual(report["written"], 1)
        self.assertEqual(report["skipped_annotated"], 1)
        again = self.store.load(ds.id)
        a, b, c = again.samples
        np.testing.assert_array_equal(again.load_index_mask(a), manual)
        self.assertEqual(a.status, "annotated")
        self.assertEqual(b.status, "predicted")
        self.assertEqual(sorted(np.unique(again.load_index_mask(b)).tolist()), [0, 1])
        self.assertEqual(c.status, "unlabeled")
        self.assertEqual(len(binary_training_pairs(again.path)), 1)
        self.assertEqual(len(binary_training_pairs(again.path, include_predicted=True)), 2)
        # 再次写回会覆盖之前的预测，但仍不触碰人工标注
        self.assertEqual(importer.write_back_predictions(again, pred)["written"], 1)

    def test_write_back_refused_for_multiclass(self) -> None:
        classes = [LabelClass(0, "背景"), LabelClass(1, "根"), LabelClass(2, "叶")]
        ds = self.store.create("多类", classes=classes)
        with self.assertRaises(DatasetProtectedError):
            importer.write_back_predictions(ds, self.base)

    def test_predictions_to_new_dataset(self) -> None:
        src, _, pred = self._binary_with_samples()
        images = [src.abs(s.image) for s in src.samples]
        ds, report = importer.predictions_to_dataset(
            self.store, images, pred, "预测结果", provenance={"source_dataset_id": src.id},
        )
        self.assertEqual(report["written"], 2)
        self.assertEqual(len(ds.samples), 3)
        self.assertEqual([s.status for s in ds.samples], ["predicted", "predicted", "unlabeled"])
        self.assertEqual(ds.meta.provenance["kind"], "predict")
        self.assertEqual(ds.meta.provenance["source_dataset_id"], src.id)
        self.assertEqual(self.store.load(src.id).samples[1].status, "unlabeled")


class TestSplits(DatasetTestCase):
    def _split_source(self, layout: str) -> Path:
        """train/val/test 各一个同名样本 x，另有 train 独有的 y。"""
        src = self.base / "src"
        for split, value in (("train", 10), ("val", 20), ("test", 30)):
            if layout == "split_first":
                img, mask = src / split / "images" / "x.png", src / split / "masks" / "x.png"
            elif layout == "images_first":
                img, mask = src / "images" / split / "x.png", src / "masks" / split / "x.png"
            else:
                img, mask = src / split / "x.png", src / split / "x_mask.png"
            _img(img, value=value)
            arr = _binary_arr()
            arr[0, 0] = 255 if split == "val" else 0  # 用像素区分各划分的掩码
            _gray_mask(mask, arr)
        _img(src / ("train/images" if layout == "split_first" else "images/train" if layout == "images_first"
                    else "train") / "y.png")
        return src

    def test_pairing_detects_splits_without_cross_split_conflicts(self) -> None:
        for layout in ("split_first", "images_first", "same_dir"):
            with self.subTest(layout=layout):
                src = self._split_source(layout)
                res = discover_pairs(src)
                self.assertEqual(res.conflicts, [])
                self.assertEqual(res.split_counts, {"train": 2, "val": 1, "test": 1})
                for p in res.pairs:
                    if p.mask is not None:
                        self.assertIn(p.split, p.mask.relative_to(src).as_posix())
                shutil.rmtree(src)

    def test_import_keeps_splits_and_layout(self) -> None:
        src = self._split_source("split_first")
        ds, report = importer.import_dataset(
            self.store, src, importer.ImportOptions(name="划分", mapping={"0": 0, "255": 1})
        )
        self.assertEqual(report["failed"], [])
        self.assertEqual(ds.split_counts(), {"": 0, "train": 2, "val": 1, "test": 1})
        for s in ds.samples:
            self.assertTrue(s.image.startswith(f"{s.split}/images/"))
            self.assertTrue(ds.abs(s.image).is_file())
            if s.mask:
                self.assertTrue(s.mask.startswith(f"{s.split}/masks/"))
        self.assertEqual(len({s.stem for s in ds.samples}), 4)
        val = next(s for s in ds.samples if s.split == "val")
        self.assertEqual(int(ds.load_index_mask(val)[0, 0]), 1)
        self.assertEqual(validation.validate_dataset(self.store.load(ds.id)), [])

        flat, _ = importer.import_dataset(
            self.store, src, importer.ImportOptions(name="不保留", mapping={"0": 0, "255": 1}, keep_splits=False)
        )
        self.assertFalse(flat.has_splits)
        self.assertTrue(all(s.image.startswith("images/") for s in flat.samples))

    def test_set_split_moves_files_and_back(self) -> None:
        ds = self.store.create("移动")
        _img(self.base / "a.png")
        s = ds.add_image(self.base / "a.png")
        ds.save_binary_annotation(s.id, _binary_arr() > 0)
        old_img, old_mask = ds.abs(s.image), ds.abs(s.mask)
        ds.set_splits([s], "val")
        again = self.store.load(ds.id)
        moved = again.samples[0]
        self.assertEqual((moved.split, moved.image, moved.mask), ("val", "val/images/a.png", "val/masks/a.png"))
        self.assertFalse(old_img.exists() or old_mask.exists())
        self.assertEqual(int(again.load_index_mask(moved).max()), 1)
        self.assertEqual(validation.validate_dataset(again), [])
        # 已划分样本重新保存标注时写到所在划分目录
        again.save_binary_annotation(moved.id, _binary_arr() > 0)
        self.assertEqual(moved.mask, "val/masks/a.png")
        again.set_splits(again.samples, None)
        back = self.store.load(ds.id).samples[0]
        self.assertEqual((back.split, back.image, back.mask), (None, "images/a.png", "masks/a.png"))

    def test_set_split_moves_instance_masks(self) -> None:
        ds = self.store.create("实例", task_type="instance")
        _img(self.base / "a.png")
        s = ds.add_image(self.base / "a.png")
        ds.set_instances(s, [(1, _binary_arr() > 0), (1, np.ones((12, 16), bool))])
        ds.set_splits([s], "test")
        self.assertTrue(all(r.mask.startswith("test/masks/a/") for r in s.instances))
        self.assertEqual(len(ds.instance_masks(s)), 2)
        self.assertFalse((ds.masks_dir / "a").exists())
        self.assertEqual(validation.validate_dataset(ds), [])

    def test_set_split_refuses_to_overwrite(self) -> None:
        ds = self.store.create("冲突")
        _img(self.base / "a.png")
        s = ds.add_image(self.base / "a.png")
        _img(ds.images_dir_for("train") / "a.png")
        with self.assertRaises(DatasetError):
            ds.set_split(s, "train")
        self.assertIsNone(s.split)
        self.assertTrue(ds.abs(s.image).is_file())

    def test_random_split_is_reproducible_and_stratified(self) -> None:
        def build(name: str):
            ds = self.store.create(name)
            for i in range(20):
                _img(self.base / "src" / f"{i:02d}.png")
                s = ds.add_image(self.base / "src" / f"{i:02d}.png")
                if i < 10:
                    ds.set_semantic_mask(s, (_binary_arr() > 0).astype(np.uint8))
            ds.save()
            return ds

        a, b = build("A"), build("B")
        counts = a.random_split({"train": 70, "val": 20, "test": 10}, seed=7)
        b.random_split({"train": 0.7, "val": 0.2, "test": 0.1}, seed=7)
        self.assertEqual(counts, {"train": 14, "val": 4, "test": 2})
        self.assertEqual([s.split for s in a.samples], [s.split for s in b.samples])
        annotated_val = sum(1 for s in a.samples if s.split == "val" and s.has_annotation)
        self.assertEqual(annotated_val, 2)
        self.assertEqual(validation.validate_dataset(self.store.load(a.id)), [])

        groups = binary_training_splits(a.path)
        self.assertEqual({k: len(v) for k, v in groups.items()}, {"": 0, "train": 7, "val": 2, "test": 1})
        self.assertEqual(len(binary_training_pairs(a.path)), 10)

    def test_random_split_tiny_dataset_fills_every_subset(self) -> None:
        ds = self.store.create("小")
        for i in range(3):
            _img(self.base / f"{i}.png")
            ds.add_image(self.base / f"{i}.png")
        self.assertEqual(ds.random_split({"train": 80, "val": 10, "test": 10}),
                         {"train": 1, "val": 1, "test": 1})

    def test_export_keeps_splits_and_reimports(self) -> None:
        src = self._split_source("split_first")
        ds, _ = importer.import_dataset(
            self.store, src, importer.ImportOptions(name="导出", mapping={"0": 0, "255": 1})
        )
        out = self.base / "export"
        convert.export_dataset(ds, out, layout="split_dir", encoding="binary")
        self.assertTrue((out / "train" / "images").is_dir())
        self.assertTrue((out / "val" / "masks").is_dir())
        self.assertFalse((out / "images").exists())
        again, _ = importer.import_dataset(
            self.store, out, importer.ImportOptions(name="再导入", mapping={"0": 0, "255": 1})
        )
        self.assertEqual(again.split_counts(), ds.split_counts())

    def test_conversion_keeps_splits(self) -> None:
        ds = self.store.create("语义")
        _img(self.base / "a.png")
        s = ds.add_image(self.base / "a.png", split="train")
        ds.set_semantic_mask(s, (_binary_arr() > 0).astype(np.uint8))
        ds.save()
        inst, _ = convert.semantic_to_instance(self.store, ds.id, "实例", min_area=1)
        self.assertEqual(inst.samples[0].split, "train")
        self.assertTrue(inst.samples[0].instances[0].mask.startswith("train/masks/"))

    @unittest.skipUnless(_HAS_TRAINING, "需要 torch / albumentations")
    def test_training_uses_dataset_splits(self) -> None:
        from plant_phenotype.training.train import build_dataloaders

        ds = self.store.create("训练")
        for i, split in enumerate(["train", "train", "train", None, "val", "test"]):
            _img(self.base / "src" / f"{i}.png", 32, 32)
            s = ds.add_image(self.base / "src" / f"{i}.png", split=split)
            ds.set_semantic_mask(s, np.pad(_binary_arr(16, 16) > 0, 8).astype(np.uint8))
        ds.save()
        cfg = {"data": {"root_dir": str(ds.path)}, "train": {"batch_size": 2, "num_workers": 0}}
        train_loader, val_loader = build_dataloaders(cfg, None, None)
        self.assertEqual(len(train_loader.dataset), 4)
        self.assertEqual(len(val_loader.dataset), 1)

    @unittest.skipUnless(_HAS_TRAINING, "需要 torch / albumentations")
    def test_training_custom_dir_with_split_subdirs(self) -> None:
        from plant_phenotype.training.train import build_dataloaders

        root = self.base / "plain"
        for split, n in (("train", 3), ("val", 2)):
            for i in range(n):
                _img(root / split / "images" / f"{split}{i}.png", 32, 32)
                _gray_mask(root / split / "masks" / f"{split}{i}_mask.png", np.pad(_binary_arr(16, 16), 8))
        cfg = {"data": {"root_dir": str(root), "images_dir": "images", "masks_dir": "masks",
                        "image_extensions": [".png"]},
               "train": {"batch_size": 2, "num_workers": 0}}
        train_loader, val_loader = build_dataloaders(cfg, None, None)
        self.assertEqual(len(train_loader.dataset), 3)
        self.assertEqual(len(val_loader.dataset), 2)


if __name__ == "__main__":
    unittest.main()
