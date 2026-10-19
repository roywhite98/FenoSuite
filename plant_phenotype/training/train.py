# -*- coding: utf-8 -*-
"""训练入口：支持 CLI 与进程内 run_training(config_path, data_root, save_dir, line_callback)。"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Tuple, Optional, Callable

warnings.filterwarnings("ignore", message=".*Error fetching version.*")
warnings.filterwarnings("ignore", message=".*version info.*")
warnings.filterwarnings("ignore", message=".*Argument.*not valid for transform.*")

import albumentations as A
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from tqdm import tqdm
import yaml

from plant_phenotype.datasets.pairing import SPLIT_ALIASES
from plant_phenotype.datasets.training_source import binary_training_splits, describe_splits, is_dataset_dir
from plant_phenotype.image_io import DEFAULT_IMAGE_EXTENSIONS, normalize_extensions
from .dataset.root_dataset import RootSegDataset
from .losses import build_criterion
from .metrics import dice_coef, iou_score
from plant_phenotype.models.build_model import build_model_from_config
from plant_phenotype.models.outputs import segmentation_logits


def set_seed(seed: int = 42) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = False
    cudnn.benchmark = True


def build_transforms(patch_size: Tuple[int, int], enable_augmentation: bool = True) -> Tuple[A.Compose, A.Compose]:
    if enable_augmentation:
        train_tf = A.Compose([
            A.RandomCrop(height=patch_size[0], width=patch_size[1]),
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
            A.Affine(translate_percent={"x": (-0.0625, 0.0625), "y": (-0.0625, 0.0625)}, scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
            A.ElasticTransform(alpha=50, sigma=10, p=0.3),
            A.RandomBrightnessContrast(p=0.5), A.GaussNoise(p=0.3),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])
    else:
        train_tf = A.Compose([
            A.RandomCrop(height=patch_size[0], width=patch_size[1]),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])
    val_tf = A.Compose([
        A.LongestMaxSize(max_size=max(patch_size)),
        A.PadIfNeeded(min_height=patch_size[0], min_width=patch_size[1], border_mode=0),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])
    return train_tf, val_tf


def _split_subdir(root: Path, split: str) -> Optional[Path]:
    """自定义数据目录下的 train / val 子目录（支持 training、valid、validation 等别名）。"""
    if not root.is_dir():
        return None
    return next(
        (d for d in sorted(root.iterdir()) if d.is_dir() and SPLIT_ALIASES.get(d.name.lower()) == split),
        None,
    )


def build_dataloaders(cfg, train_tf, val_tf):
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    root_dir = data_cfg.get("root_dir") or "./workspace/datasets/data"
    images_dir = data_cfg.get("images_dir", "")
    masks_dir = data_cfg.get("masks_dir", "")
    mask_suffix = data_cfg.get("mask_suffix", "_mask")
    mask_ext = data_cfg.get("mask_ext", ".png")
    # 支持多种图像格式：配置里可写 image_extensions: [.png, .jpg, .tif, ...] 或仅 image_ext: .tif
    image_extensions_cfg = data_cfg.get("image_extensions")
    if image_extensions_cfg is not None:
        image_extensions = normalize_extensions(image_extensions_cfg)
    else:
        image_ext = data_cfg.get("image_ext")
        image_extensions = normalize_extensions(image_ext) if image_ext else DEFAULT_IMAGE_EXTENSIONS
    common = dict(
        images_dir=images_dir, masks_dir=masks_dir,
        image_ext=image_extensions[0] if image_extensions else ".png",
        image_extensions=image_extensions,
        mask_ext=mask_ext, mask_suffix=mask_suffix,
    )
    # 已有划分时：训练集（数据集中的未划分样本也并入）用于训练，验证集用于验证，测试集不参与；
    # 没有验证集时再按 train_val_split 从训练样本中随机划出
    pairs = val_pairs = None
    train_root, val_root = root_dir, None
    if is_dataset_dir(root_dir):
        groups = binary_training_splits(root_dir, include_predicted=bool(data_cfg.get("include_predicted", False)))
        logging.info(f"从数据集索引读取已标注样本: {root_dir}；{describe_splits(groups)}")
        pairs = groups["train"] + groups[""]
        val_pairs = groups["val"] or None
        if not pairs:
            raise RuntimeError("数据集中没有训练集或未划分的已标注样本（验证集、测试集样本不用于训练）")
    else:
        train_dir = _split_subdir(Path(root_dir), "train")
        if train_dir is not None:
            val_dir = _split_subdir(Path(root_dir), "val")
            train_root, val_root = str(train_dir), (str(val_dir) if val_dir else None)
            logging.info(f"数据目录按划分组织：训练 {train_dir}，验证 {val_dir or '（无，将随机划出）'}")
    dataset = RootSegDataset(root_dir=train_root, **common, transform=None, pairs=pairs)
    if val_pairs is not None or val_root is not None:
        train_sel = dict(root_dir=train_root, pairs=pairs) if pairs is not None \
            else dict(root_dir=train_root, file_list=dataset.image_files)
        val_sel = dict(root_dir=root_dir, pairs=val_pairs) if val_pairs is not None else dict(root_dir=val_root)
    else:
        n_total = len(dataset)
        n_train = max(1, min(n_total - 1, int(n_total * data_cfg.get("train_val_split", 0.8))))
        n_val = n_total - n_train
        train_subset, val_subset = random_split(dataset, [n_train, n_val])
        if pairs is not None:
            train_sel = dict(root_dir=train_root, pairs=[pairs[i] for i in train_subset.indices])
            val_sel = dict(root_dir=train_root, pairs=[pairs[i] for i in val_subset.indices])
        else:
            train_sel = dict(root_dir=train_root, file_list=[dataset.image_files[i] for i in train_subset.indices])
            val_sel = dict(root_dir=train_root, file_list=[dataset.image_files[i] for i in val_subset.indices])
    use_weight_mask = cfg.get("loss", {}).get("use_weight_mask", False)
    foreground_weight = cfg.get("loss", {}).get("foreground_weight", 3.0)
    oversample_fg = train_cfg.get("oversample_foreground", 1.0)
    undersample_bg = train_cfg.get("undersample_background", 1.0)
    train_dataset = RootSegDataset(
        **common, **train_sel,
        transform=train_tf, return_weight_mask=use_weight_mask, foreground_weight=foreground_weight,
    )
    val_dataset = RootSegDataset(**common, **val_sel, transform=val_tf)
    logging.info(f"训练样本 {len(train_dataset)} 个，验证样本 {len(val_dataset)} 个")
    batch_size = train_cfg["batch_size"]
    num_workers = train_cfg.get("num_workers", 4)

    # 8.1: num_workers 安全回退——Windows 或配置为 0 时强制单进程
    if num_workers == 0 or sys.platform == "win32":
        num_workers = 0
        logging.info("DataLoader num_workers 已设为 0（单进程模式）")
    else:
        logging.info(f"DataLoader num_workers={num_workers}")

    # 8.2: batch_size 自动调整——数据集样本数不足时缩小 batch_size
    if len(train_dataset) < batch_size:
        logging.warning(
            f"训练集样本数 ({len(train_dataset)}) 小于 batch_size ({batch_size})，"
            f"已自动将 batch_size 调整为 {len(train_dataset)}"
        )
        batch_size = len(train_dataset)

    # 8.5: pin_memory 自动修正——非 CUDA 设备时关闭 pin_memory
    pin_memory = True
    if pin_memory and not torch.cuda.is_available():
        logging.warning("当前设备不支持 CUDA，已自动将 pin_memory 设为 False")
        pin_memory = False

    drop_last = len(train_dataset) > batch_size
    sampler = None
    shuffle = True
    if oversample_fg > 1.0 or undersample_bg < 1.0:
        weights = train_dataset.get_sample_weights(
            oversample_foreground=max(1e-6, float(oversample_fg)),
            undersample_background=max(1e-6, float(undersample_bg)),
        )
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        shuffle = False
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=pin_memory, drop_last=False)
    return train_loader, val_loader


def build_optimizer_scheduler(model, cfg, n_iterations: int, n_epochs: int):
    train_cfg = cfg["train"]
    lr = train_cfg["lr"]
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("没有可训练参数，请检查 freeze_encoder_stages")
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    schedule_type = train_cfg.get("lr_schedule", "poly")
    if schedule_type == "poly":
        power = train_cfg.get("poly_power", 0.9)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda it: (1 - it / float(max(1, n_iterations))) ** power)
    elif schedule_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iterations)
    elif schedule_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=train_cfg.get("step_size", 30), gamma=train_cfg.get("step_gamma", 0.1))
    elif schedule_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=train_cfg.get("plateau_factor", 0.5),
            patience=train_cfg.get("plateau_patience", 10),
        )
    else:
        scheduler = None
    return optimizer, scheduler


def _step_scheduler(scheduler, val_dice: float, batches_per_epoch: int) -> None:
    """Preserve the existing epoch-end stepping convention for every schedule."""
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(val_dice)
    elif isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
        scheduler.step()
    else:
        for _ in range(batches_per_epoch):
            scheduler.step()


def _restore_scheduler_state(scheduler, checkpoint: dict, batches_per_epoch: int,
                             log: Callable[[str], None] = logging.warning) -> None:
    """Restore after loading the optimizer, including pre-scheduler checkpoints."""
    if scheduler is None:
        return
    state = checkpoint.get("scheduler_state")
    if state is not None:
        scheduler.load_state_dict(state)
        return

    completed_epochs = int(checkpoint.get("epoch", 0))
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.last_epoch = completed_epochs
        if "best_dice" in checkpoint:
            scheduler.best = float(checkpoint["best_dice"])
        log("  旧 checkpoint 缺少 scheduler 状态：保留学习率和最佳 Dice，"
            "Plateau 的等待轮次及冷却计数从 0 开始，无法精确恢复历史调度")
    else:
        steps = completed_epochs if isinstance(scheduler, torch.optim.lr_scheduler.StepLR) \
            else completed_epochs * batches_per_epoch
        # The optimizer already holds the LR *after* this epoch's scheduler
        # steps. Move only the counters; replaying steps would decay it twice.
        scheduler.last_epoch = steps
        scheduler._step_count = steps + 1
        scheduler.base_lrs = [group.get("initial_lr", base_lr) for group, base_lr in
                              zip(scheduler.optimizer.param_groups, scheduler.base_lrs)]
        log(f"  旧 checkpoint 缺少 scheduler 状态，按已完成 {completed_epochs} 轮恢复至调度步 {steps}")
    scheduler._last_lr = [group["lr"] for group in scheduler.optimizer.param_groups]


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, max_grad_norm: float, log_interval: int,
                    epoch: int, total_epochs: int, use_weight_mask: bool = False):
    model.train()
    running_loss = 0.0
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Train {epoch}/{total_epochs}")
    for step, batch in pbar:
        if use_weight_mask and len(batch) == 3:
            images, masks, weight_masks = batch
            weight_masks = weight_masks.to(device, non_blocking=True)
        else:
            images, masks = batch[:2]
            weight_masks = None
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=scaler is not None and scaler.is_enabled()):
            logits = model(images)
            loss = criterion(logits, masks, weight_masks)
        if not torch.isfinite(loss).all():
            continue
        if scaler is not None:
            scaler.scale(loss).backward()
            if max_grad_norm and max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        running_loss += loss.item()
        if (step + 1) % log_interval == 0:
            pbar.set_postfix(loss=running_loss / (step + 1))
    return running_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device, use_weight_mask: bool = False):
    model.eval()
    if len(loader) == 0:
        return 0.0, 0.0, 0.0
    val_loss = 0.0
    dices, ious = [], []
    for batch in tqdm(loader, desc="Val", total=len(loader)):
        if use_weight_mask and len(batch) == 3:
            images, masks, weight_masks = batch
            weight_masks = weight_masks.to(device, non_blocking=True)
        else:
            images, masks = batch[:2]
            weight_masks = None
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        outputs = model(images)
        val_loss += criterion(outputs, masks, weight_masks).item()
        logits = segmentation_logits(outputs)
        dices.append(dice_coef(logits, masks))
        ious.append(iou_score(logits, masks))
    val_loss /= len(loader)
    return val_loss, float(np.mean(dices)), float(np.mean(ious))


def _run_training_impl(
    cfg: dict,
    line_callback: Optional[Callable[[str], None]] = None,
    use_cpu: bool = False,
    resume_from: Optional[str] = None,
) -> None:
    def log(msg: str) -> None:
        if line_callback:
            line_callback(msg)
        else:
            print(msg)

    set_seed(42)
    if use_cpu:
        device = torch.device("cpu")
        log("使用 CPU 训练（已勾选仅使用 CPU）")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu":
            log("未检测到 CUDA，将使用 CPU 训练（速度较慢属正常）")
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    loss_cfg = cfg["loss"]
    patch_size = tuple(train_cfg.get("patch_size", [512, 512]))
    train_tf, val_tf = build_transforms(patch_size=patch_size, enable_augmentation=train_cfg.get("enable_augmentation", True))
    train_loader, val_loader = build_dataloaders(cfg, train_tf, val_tf)

    name_lower = model_cfg.get("name", "").lower()
    if name_lower == "dinov3_mlp":
        try:
            import transformers  # noqa: F401
        except ImportError:
            log("Error: DINOv3 模型需要 'transformers'。请安装: pip install transformers accelerate")
            raise SystemExit(1)
    init_checkpoint = None
    if name_lower in ("mobile_sam", "mobilesam") and not resume_from:
        init_checkpoint = model_cfg.get("checkpoint")
        if not init_checkpoint or not Path(str(init_checkpoint)).is_file():
            log("Error: 请在 model.checkpoint 填写官方 mobile_sam.pt 的路径（见 resources/configs/mobile_sam.yaml）")
            raise SystemExit(1)
        init_checkpoint = str(init_checkpoint)

    model = build_model_from_config(model_cfg, checkpoint=init_checkpoint).to(device)
    frozen_stages = [str(s) for s in (model_cfg.get("freeze_encoder_stages") or [])]
    if frozen_stages:
        frozen_n = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        trainable_n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"已冻结编码器阶段 {frozen_stages}（含 BN 运行统计，stem 仍可训练）：冻结 {frozen_n}，可训练 {trainable_n}")
    criterion = build_criterion(loss_cfg, device)
    total_epochs = train_cfg["epochs"]
    n_iterations = total_epochs * len(train_loader)
    optimizer, scheduler = build_optimizer_scheduler(model, cfg, n_iterations, total_epochs)
    amp_flag = train_cfg.get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_flag)
    save_dir = Path(train_cfg.get("save_dir", "./workspace/runs/training"))
    save_dir.mkdir(parents=True, exist_ok=True)
    log_interval = train_cfg.get("log_interval", 10)
    max_grad_norm = train_cfg.get("max_grad_norm", 12.0)
    early_stop_patience = train_cfg.get("early_stop_patience", 20)
    enable_visualization = train_cfg.get("enable_visualization", True)
    best_dice = -1.0
    start_epoch = 1
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": [], "lr": []}

    if resume_from and Path(resume_from).is_file():
        log(f"从 checkpoint 继续训练: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        state = ckpt.get("model_state") or ckpt.get("state_dict") or ckpt
        load_result = model.load_state_dict(state, strict=False)
        missing = getattr(load_result, "missing_keys", []) or []
        unexpected = getattr(load_result, "unexpected_keys", []) or []
        if missing:
            log(f"  加载时缺少的键（将随机初始化）: {len(missing)} 个")
        if unexpected:
            log(f"  加载时多余的键（已忽略）: {len(unexpected)} 个")
        if "optimizer_state" in ckpt and "epoch" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            except Exception as e:
                log(f"  恢复优化器/轮次失败，仅使用模型权重: {e}")
            else:
                start_epoch = int(ckpt["epoch"]) + 1
                best_dice = float(ckpt.get("best_dice", -1.0))
                _restore_scheduler_state(scheduler, ckpt, len(train_loader), log)
                if ckpt.get("scaler_state") and scaler.is_enabled():
                    scaler.load_state_dict(ckpt["scaler_state"])
                log(f"  已恢复优化器、调度器与轮次，从 epoch {start_epoch} 继续，当前 best_dice={best_dice:.4f}")
        else:
            log("  checkpoint 无优化器/轮次，仅加载模型参数，从第 1 轮开始")
    elif resume_from:
        log(f"警告: 继续训练路径不存在或不是文件，将从头训练: {resume_from}")

    for epoch in range(start_epoch, total_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            max_grad_norm, log_interval, epoch, total_epochs,
            use_weight_mask=loss_cfg.get("use_weight_mask", False),
        )
        val_loss, val_dice, val_iou = validate(
            model, val_loader, criterion, device,
            use_weight_mask=loss_cfg.get("use_weight_mask", False),
        )
        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)
        history["lr"].append(current_lr)
        _step_scheduler(scheduler, val_dice, len(train_loader))
        log(f"Epoch [{epoch}/{total_epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_dice={val_dice:.4f} val_iou={val_iou:.4f} lr={current_lr:.6f}")
        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            best_path = save_dir / "best_model.pth"
            torch.save({
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "scaler_state": scaler.state_dict(), "epoch": epoch,
                "best_dice": best_dice, "config": cfg,
            }, best_path)
            log(f"Saved best model to {best_path} (dice={best_dice:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                log(f"Early stopping at epoch {epoch} (patience={early_stop_patience})")
                break

    if enable_visualization:
        try:
            from .visualize import plot_training_curves, plot_confusion_style_metrics, save_training_log
            save_training_log(history, cfg, save_dir)
            plot_training_curves(history, str(save_dir / "training_curves.png"), title=cfg.get("experiment_name", "Training"))
            best_iou = max(history["val_iou"]) if history["val_iou"] else 0.0
            plot_confusion_style_metrics(best_dice, best_iou, str(save_dir / "final_metrics.png"))
            log("训练可视化和日志已生成！")
        except Exception as e:
            log(f"可视化失败（非致命）：{e}")


def run_training(
    config_path: str,
    data_root: Optional[str] = None,
    save_dir: Optional[str] = None,
    line_callback: Optional[Callable[[str], None]] = None,
    use_cpu: bool = False,
    resume_from: Optional[str] = None,
) -> None:
    """进程内执行训练，供 GUI/ server 调用。"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if data_root is not None:
        cfg.setdefault("data", {})["root_dir"] = str(data_root)
    if save_dir is not None:
        cfg.setdefault("train", {})["save_dir"] = str(save_dir)
    _run_training_impl(cfg, line_callback, use_cpu=use_cpu, resume_from=resume_from)


def main() -> None:
    parser = argparse.ArgumentParser(description="Root segmentation training (nnU-Net style 2D)")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "resources" / "configs" / "unet.yaml"),
    )
    parser.add_argument("--data-root", type=str, default=None, help="覆盖 data.root_dir")
    parser.add_argument("--save-dir", type=str, default=None, help="覆盖 train.save_dir")
    parser.add_argument("--resume-from", type=str, default=None, help="继续训练的 .pth 权重")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_training(
        config_path=args.config,
        data_root=args.data_root,
        save_dir=args.save_dir,
        use_cpu=args.cpu,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
