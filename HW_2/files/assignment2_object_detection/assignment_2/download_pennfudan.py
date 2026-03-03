"""
Download and prepare the Penn-Fudan Pedestrian Dataset.

Usage:
    python data/download_pennfudan.py

Downloads to: data/PennFudanPed/
Splits into:  data/pennfudan/{train, val, test}/
"""

import os
import shutil
import random
import zipfile
import urllib.request
from pathlib import Path
from tqdm import tqdm


URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
DATA_ROOT = Path("data")
RAW_DIR   = DATA_ROOT / "PennFudanPed"
OUT_DIR   = DATA_ROOT / "pennfudan"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 0.15  (remainder)

SEED = 42


class _DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} …")
    with _DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=dest.name) as t:
        urllib.request.urlretrieve(url, dest, reporthook=t.update_to)


def extract(zip_path: Path, out_dir: Path) -> None:
    print(f"Extracting to {out_dir} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def split_dataset(raw_dir: Path, out_dir: Path, seed: int = SEED) -> None:
    """Split PNGImages into train/val/test directories preserving paired masks."""
    img_dir  = raw_dir / "PNGImages"
    mask_dir = raw_dir / "PedMasks"

    images = sorted(img_dir.glob("*.png"))
    random.seed(seed)
    random.shuffle(images)

    n      = len(images)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    splits = {
        "train": images[:n_train],
        "val":   images[n_train: n_train + n_val],
        "test":  images[n_train + n_val:],
    }

    for split_name, split_imgs in splits.items():
        for subdir in ("images", "masks"):
            (out_dir / split_name / subdir).mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(split_imgs, desc=f"Copying {split_name}"):
            mask_path = mask_dir / img_path.name.replace(".png", "_mask.png")
            shutil.copy(img_path,  out_dir / split_name / "images" / img_path.name)
            if mask_path.exists():
                shutil.copy(mask_path, out_dir / split_name / "masks" / mask_path.name)

    print("\nSplit summary:")
    for split_name, split_imgs in splits.items():
        print(f"  {split_name:5s}: {len(split_imgs):3d} images")


def main():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    zip_path = DATA_ROOT / "PennFudanPed.zip"

    # --- Download ---
    if not zip_path.exists():
        download(URL, zip_path)
    else:
        print(f"Zip already exists at {zip_path}, skipping download.")

    # --- Extract ---
    if not RAW_DIR.exists():
        extract(zip_path, DATA_ROOT)
    else:
        print(f"Raw data already extracted at {RAW_DIR}, skipping.")

    # --- Split ---
    if OUT_DIR.exists():
        print(f"Split already exists at {OUT_DIR}, skipping split.")
    else:
        split_dataset(RAW_DIR, OUT_DIR)

    print(f"\nPenn-Fudan dataset ready at: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
