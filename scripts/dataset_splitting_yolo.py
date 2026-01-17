import random
import shutil
from pathlib import Path


def split_yolo_dataset(
    src_dir="dataset_drones",
    dst_dir="dataset_drones_split",
    train_ratio=0.70,
    val_ratio=0.20,
    test_ratio=0.10,
    seed=42,
    copy=True
):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    img_src = src_dir / "images"
    lbl_src = src_dir / "labels"

    imgs = sorted(list(img_src.glob("*.jpg")) + list(img_src.glob("*.jpeg")) + list(img_src.glob("*.png")))
    if not imgs:
        raise RuntimeError(f"No images found in {img_src}")

      # filter: keep only pairs that have label
    pairs = []
    missing = 0
    for img in imgs:
        lbl = lbl_src / (img.stem + ".txt")
        if lbl.exists():
            pairs.append((img, lbl))
        else:
            missing += 1
            
    if not pairs:
        raise RuntimeError("No image/label pairs found.")
    if missing:
        print(f"⚠️ Missing labels for {missing} images (skipped).")

    # sanity: ratios sum
    s = train_ratio + val_ratio + test_ratio
    if abs(s - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")
    
    random.seed(seed)
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val

    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }

    # create folders
    for split in ["train", "val", "test"]:
        (dst_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    op = shutil.copy2 if copy else shutil.move

    for split, items in splits.items():
        for img, lbl in items:
            op(img, dst_dir / "images" / split / img.name)
            op(lbl, dst_dir / "labels" / split / lbl.name)

    print(f"Done. Total pairs: {n}")
    print(f"train: {len(splits['train'])}, val: {len(splits['val'])}, test: {len(splits['test'])}")
    print(f"Output: {dst_dir.resolve()}")

if __name__ == "__main__":
    split_yolo_dataset(src_dir="dataset_drones", dst_dir="dataset_drones_split", seed=42, copy=True)
