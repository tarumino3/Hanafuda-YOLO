#!/usr/bin/env python3
"""Training script for Hanafuda card detection using YOLO11n.

Usage
-----
    python -m src.train --help
    python -m src.train --data-dir data/raw --epochs 100 --device 0
    python -m src.train --data-dir data/raw --val-split 0.2 --seed 42 --no-wandb
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from .utils import CLASS_NAMES, TrainConfig, setup_wandb


def prepare_dataset(data_dir: Path, val_split: float, seed: int) -> Path:
    """Split a flat dataset into train/val and generate dataset.yaml.

    If ``images/train/`` already exists and is non-empty, the split is
    skipped and the existing ``dataset.yaml`` is returned — so re-running
    with the same seed is idempotent.

    Args:
        data_dir: Root directory containing ``images/``, ``labels/``,
                  and ``classes.txt``.
        val_split: Fraction of images to use for validation (e.g. ``0.2``).
        seed: Random seed for the split — same seed always gives the same split.

    Returns:
        Path to the generated (or existing) ``dataset.yaml``.
    """
    from sklearn.model_selection import train_test_split  # noqa: PLC0415
    import yaml  # noqa: PLC0415

    train_img_dir = data_dir / "images" / "train"
    yaml_path = data_dir / "dataset.yaml"

    # Idempotent: skip if split already exists
    if train_img_dir.exists() and any(train_img_dir.iterdir()) and yaml_path.exists():
        print(f"[dataset] Using existing split in {data_dir}")
        return yaml_path

    # Collect all images from the flat images/ directory
    image_dir = data_dir / "images" / "raw"
    label_dir = data_dir / "labels" / "raw"
    images = sorted(
        p
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")

    train_imgs, val_imgs = train_test_split(
        images, test_size=val_split, random_state=seed
    )
    print(
        f"[dataset] Split: {len(train_imgs)} train / {len(val_imgs)} val  (seed={seed})"
    )

    for split_name, split_imgs in [("train", train_imgs), ("valid", val_imgs)]:
        out_img = data_dir / "images" / split_name
        out_lbl = data_dir / "labels" / split_name
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        for img_path in split_imgs:
            shutil.copy2(img_path, out_img / img_path.name)
            lbl_path = label_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                shutil.copy2(lbl_path, out_lbl / lbl_path.name)

    # Read class names from classes.txt; fall back to CLASS_NAMES from utils
    classes_file = data_dir / "classes.txt"
    if classes_file.exists():
        class_names = [
            line.strip()
            for line in classes_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        print("[dataset] classes.txt not found — using CLASS_NAMES from src/utils.py")
        class_names = CLASS_NAMES

    yaml_content = {
        "path": str(data_dir.resolve()),
        "train": "images/train",
        "val": "images/valid",
        "nc": len(class_names),
        "names": class_names,
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False)

    print(f"[dataset] Generated {yaml_path}  (nc={len(class_names)})")
    return yaml_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLO11n model on the Hanafuda dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", default="yolo11n.pt", help="Base YOLO model or checkpoint path"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        dest="data_dir",
        help="Root directory of the raw dataset (must contain images/ and labels/)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        dest="val_split",
        help="Fraction of data held out for validation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/val split",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help='"0" for GPU, "cpu" for CPU')
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--patience", type=int, default=20, help="Early stopping patience (0 = off)"
    )
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="hanafuda_yolo11n")
    parser.add_argument(
        "--wandb-project", default="hanafuda-yolo", dest="wandb_project"
    )
    parser.add_argument("--wandb-entity", default="", dest="wandb_entity")
    parser.add_argument(
        "--no-wandb",
        action="store_false",
        dest="use_wandb",
        help="Disable Weights & Biases logging",
    )
    parser.set_defaults(use_wandb=True)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainConfig:
    """Convert parsed CLI args into a TrainConfig dataclass."""
    return TrainConfig(
        model=args.model,
        data_dir=args.data_dir,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=args.project,
        name=args.name,
        val_split=args.val_split,
        seed=args.seed,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )


def train(config: TrainConfig, use_wandb: bool = True) -> None:
    """Prepare the dataset split, then run YOLO11n training.

    The ``ultralytics`` import is deferred so that importing this module
    (e.g. in tests or notebooks) does not trigger CUDA initialisation.
    """
    from ultralytics import YOLO  # noqa: PLC0415

    # 1. Create train/val split and generate dataset.yaml
    yaml_path = prepare_dataset(
        Path(config.data_dir),
        val_split=config.val_split,
        seed=config.seed,
    )
    config.data_yaml = str(yaml_path)

    # 2. Optionally initialise WandB
    wandb_active = use_wandb and setup_wandb(config)

    # 3. Train
    model = YOLO(config.model)
    model.train(**config.as_yolo_kwargs())

    if wandb_active:
        try:
            import wandb  # noqa: PLC0415

            wandb.finish()
        except ImportError:
            pass


def main() -> None:
    args = parse_args()
    config = build_config(args)

    print(f"Training config:\n{config}\n")

    try:
        train(config, use_wandb=args.use_wandb)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
