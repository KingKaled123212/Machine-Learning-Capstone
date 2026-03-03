"""
Download and prepare a 7-breed subset of the Oxford-IIIT Pet Dataset.

Usage:
    python data/download_oxford_pets.py [--breeds beagle bengal ...] [--max-per-class 72]

Downloads to: data/oxford-iiit-pet/
Splits into:  data/oxford_pets/{train, val, test}/
Also creates: data/oxford_pets/dataset.yaml  (YOLO-format class config)
"""

import os
import shutil
import random
import tarfile
import argparse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


IMAGES_URL      = "https://thor.robots.ox.ac.uk/datasets/pets/images.tar.gz"
ANNOTATIONS_URL = "https://thor.robots.ox.ac.uk/datasets/pets/annotations.tar.gz"

DATA_ROOT = Path("data")
RAW_DIR   = DATA_ROOT / "oxford-iiit-pet"
OUT_DIR   = DATA_ROOT / "oxford_pets"

# Default 7-breed subset: mix of cats and dogs with visual diversity
DEFAULT_BREEDS = [
    "Beagle",
    "Bengal",
    "British_Shorthair",
    "German_Shepherd",
    "golden_retriever",
    "Persian",
    "Pug",
]

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
SEED        = 42


class _ProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  {dest.name} already downloaded, skipping.")
        return
    print(f"Downloading {url} …")
    with _ProgressBar(unit="B", unit_scale=True, miniters=1, desc=dest.name) as t:
        urllib.request.urlretrieve(url, dest, reporthook=t.update_to)


def extract_tar(tar_path: Path, out_dir: Path) -> None:
    print(f"Extracting {tar_path.name} …")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(out_dir)


def parse_annotation(xml_path: Path):
    """Return (xmin, ymin, xmax, ymax) from a VOC-format annotation XML."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    obj = root.find("object")
    if obj is None:
        return None
    bb = obj.find("bndbox")
    if bb is None:
        return None
    return (
        int(bb.find("xmin").text),
        int(bb.find("ymin").text),
        int(bb.find("xmax").text),
        int(bb.find("ymax").text),
    )


def build_subset(
    raw_dir: Path,
    out_dir: Path,
    breeds: list,
    max_per_class: int,
    seed: int = SEED,
) -> dict:
    """
    Collect up to max_per_class images per breed, then split into train/val/test.
    Returns class_to_idx mapping.
    """
    img_dir  = raw_dir / "images"
    ann_dir  = raw_dir / "annotations" / "xmls"

    # Normalize breed names for case-insensitive matching
    breed_set = {b.lower() for b in breeds}

    # Group images by breed
    breed_images = defaultdict(list)
    for img_path in sorted(img_dir.glob("*.jpg")):
        # Oxford Pets naming: breed_name_NNN.jpg (breed may be multi-word with _)
        # The breed is everything except the trailing _NNN part
        stem_parts = img_path.stem.rsplit("_", 1)
        if len(stem_parts) < 2:
            continue
        breed_name = stem_parts[0]
        if breed_name.lower() in breed_set:
            breed_images[breed_name].append(img_path)

    if not breed_images:
        raise RuntimeError(
            "No matching breed images found. Check that breed names match the "
            "Oxford Pets naming convention (e.g. 'Beagle', 'Bengal', 'Persian')."
        )

    # Build class index
    found_breeds = sorted(breed_images.keys())
    class_to_idx = {b: i for i, b in enumerate(found_breeds)}
    print(f"\nFound {len(found_breeds)} breeds: {found_breeds}")

    random.seed(seed)

    # Collect samples per split
    splits = {"train": [], "val": [], "test": []}

    for breed, imgs in breed_images.items():
        random.shuffle(imgs)
        imgs = imgs[:max_per_class]
        n      = len(imgs)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)

        splits["train"].extend([(p, breed) for p in imgs[:n_train]])
        splits["val"].extend(  [(p, breed) for p in imgs[n_train: n_train + n_val]])
        splits["test"].extend( [(p, breed) for p in imgs[n_train + n_val:]])

    # Copy files + annotations to output directories
    for split_name, items in splits.items():
        img_out = out_dir / split_name / "images"
        ann_out = out_dir / split_name / "annotations"
        img_out.mkdir(parents=True, exist_ok=True)
        ann_out.mkdir(parents=True, exist_ok=True)

        for img_path, breed in tqdm(items, desc=f"Copying {split_name:5s}"):
            shutil.copy(img_path, img_out / img_path.name)
            # Copy XML annotation if available
            xml_path = ann_dir / (img_path.stem + ".xml")
            if xml_path.exists():
                shutil.copy(xml_path, ann_out / xml_path.name)

    print("\nSubset split summary:")
    for split_name, items in splits.items():
        print(f"  {split_name:5s}: {len(items):3d} images")

    return class_to_idx


def write_yaml(out_dir: Path, class_to_idx: dict) -> None:
    """Write a YOLO-compatible dataset.yaml."""
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    names = [idx_to_class[i] for i in range(len(idx_to_class))]

    yaml_content = f"""# Oxford Pets subset — YOLO dataset config
path: {out_dir.resolve()}
train: train/images
val:   val/images
test:  test/images

nc: {len(names)}
names: {names}
"""
    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(yaml_content)
    print(f"\nYOLO dataset config written to {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="Download & prepare Oxford Pets subset")
    parser.add_argument(
        "--breeds", nargs="+", default=DEFAULT_BREEDS,
        help="List of breed names to include (default: 7-breed subset)"
    )
    parser.add_argument(
        "--max-per-class", type=int, default=72,
        help="Max images per breed class (default: 72)"
    )
    args = parser.parse_args()

    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    img_tar  = DATA_ROOT / "images.tar.gz"
    ann_tar  = DATA_ROOT / "annotations.tar.gz"

    # --- Download ---
    download(IMAGES_URL,      img_tar)
    download(ANNOTATIONS_URL, ann_tar)

    # --- Extract ---
    if not (RAW_DIR / "images").exists():
        extract_tar(img_tar, RAW_DIR)
    else:
        print("Images already extracted, skipping.")

    if not (RAW_DIR / "annotations").exists():
        extract_tar(ann_tar, RAW_DIR)
    else:
        print("Annotations already extracted, skipping.")

    # --- Build subset ---
    if OUT_DIR.exists():
        print(f"\nSubset already exists at {OUT_DIR}. Delete it to rebuild.")
    else:
        class_to_idx = build_subset(RAW_DIR, OUT_DIR, args.breeds, args.max_per_class)
        write_yaml(OUT_DIR, class_to_idx)

    print(f"\nOxford Pets subset ready at: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
