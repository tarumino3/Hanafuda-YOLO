# Dataset

## Overview

The dataset contains **300 annotated images** of Hanafuda (花札) playing cards in YOLO format,
hosted on Hugging Face.

| Property | Value |
|----------|-------|
| Total images | 300 |
| Total annotations | 2,232 |
| Avg labels per image | 7.44 |
| Classes | 36 |
| Format | YOLO (`class_id x_center y_center width height`, normalised) |
| License | CC BY 4.0 |

Card type prefixes: `hkr` = 光 (hikari) · `tne` = 種 (tane) · `tan` = 短冊 (tanzaku) · `kas` = カス (kasu)

> **Note on class imbalance:** The dataset reflects natural Hanafuda gameplay distribution.
> "Kasu" (plain) cards appear with higher frequency because most months contain two kasu cards.
> The images include overlapping cards, hand-held cards, and scattered layouts to simulate
> actual inference conditions.

---

## Download from Hugging Face

```bash
pip install huggingface_hub

hf download tarumino3/Hanafuda-Object-Detection --repo-type dataset --local-dir data
```

Dataset page: <https://huggingface.co/datasets/tarumino3/Hanafuda-Object-Detection>

### Directory structure after download

```
data/raw/
├── images/raw       # 300 source images (.JPG)
├── labels/raw       # 300 YOLO annotation files (.txt)
├── classes.txt      # Class index mapping (36 classes)
└── notes.json       # Export metadata
```

---

## Train/Val Split

There is no predefined split in the raw dataset.
`train.py` handles it automatically — just pass `--data-dir`:

```bash
python -m src.train --data-dir data --val-split 0.2 --seed 42
```

This creates `data/raw/images/train/`, `data/raw/images/valid/`, and
`data/raw/dataset.yaml` automatically before training starts.
The same `--seed` value always produces the same split, ensuring reproducibility.

---

## dataset.yaml (auto-generated)

`train.py` generates this file from `classes.txt`. Its contents look like this:

```yaml
path: data/raw
train: images/train
val: images/valid

nc: 36
names:
  - 01-hkr-tsuru
  - 01-kas
  - 01-tan-akatan
  - 02-kas
  - 02-tan-akatan
  - 02-tne-uguisu
  - 03-hkr-maku
  - 03-kas
  - 03-tan-akatan
  - 04-kas
  - 04-tan-muji
  - 04-tne-hototogisu
  - 05-kas
  - 05-tan-muji
  - 05-tne-yatsuhashi
  - 06-kas
  - 06-tan-aotan
  - 06-tne-cho
  - 07-kas
  - 07-tan-muji
  - 07-tne-inoshishi
  - 08-hkr-tsuki
  - 08-kas
  - 08-tne-kari
  - 09-kas
  - 09-tan-aotan
  - 09-tne-sakazuki
  - 10-kas
  - 10-tan-aotan
  - 10-tne-shika
  - 11-hkr-michikaze
  - 11-kas
  - 11-tan-muji
  - 11-tne-tsubame
  - 12-hkr-hooh
  - 12-kas
```
