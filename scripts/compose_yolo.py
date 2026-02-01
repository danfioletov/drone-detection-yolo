import os
import glob
import random
from pathlib import Path

import numpy as np
import cv2


# -------------------------
# Utils
# -------------------------

def list_images(folder, exts=("png", "jpg", "jpeg")):
    files = []
    for e in exts:
        files += glob.glob(os.path.join(folder, f"*.{e}"))
        files += glob.glob(os.path.join(folder, f"*.{e.upper()}"))
    return sorted(files)

def read_sprite_rgba(path):
    """Read RGBA sprite as BGRA (OpenCV default) with alpha."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read sprite: {path}")
    if img.ndim != 3 or img.shape[2] != 4:
        raise RuntimeError(f"Sprite must be RGBA/4ch PNG: {path}")
    return img  # BGRA

def alpha_bbox(bgra, alpha_thresh=8):
    """Return bbox (x1,y1,x2,y2) from alpha channel or None."""
    a = bgra[:, :, 3]
    ys, xs = np.where(a > alpha_thresh)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return x1, y1, x2, y2

def crop_to_bbox(bgra, bbox, pad=2):
    x1, y1, x2, y2 = bbox
    h, w = bgra.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad)
    y2 = min(h - 1, y2 + pad)
    return bgra[y1:y2+1, x1:x2+1]

def resize_keep_aspect(img, scale):
    h, w = img.shape[:2]
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

def overlay_bgra_on_bgr(bg, fg_bgra, x, y):
    """Alpha-blend fg onto bg at top-left (x,y). bg is BGR, fg is BGRA."""
    bh, bw = bg.shape[:2]
    fh, fw = fg_bgra.shape[:2]

    # Clip to background
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bw, x + fw)
    y2 = min(bh, y + fh)
    if x1 >= x2 or y1 >= y2:
        return bg, None  # fully out of frame

    fg_x1 = x1 - x
    fg_y1 = y1 - y
    fg_x2 = fg_x1 + (x2 - x1)
    fg_y2 = fg_y1 + (y2 - y1)

    roi = bg[y1:y2, x1:x2].astype(np.float32)
    fg = fg_bgra[fg_y1:fg_y2, fg_x1:fg_x2].astype(np.float32)

    alpha = fg[:, :, 3:4] / 255.0
    fg_rgb = fg[:, :, :3]

    out = roi * (1 - alpha) + fg_rgb * alpha
    bg[y1:y2, x1:x2] = out.astype(np.uint8)

    # bbox of placed fg region (in bg coords), but we will refine via alpha later
    return bg, (x1, y1, x2 - 1, y2 - 1)

def yolo_from_bbox(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox
    bw = (x2 - x1 + 1)
    bh = (y2 - y1 + 1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return (cx / img_w, cy / img_h, bw / img_w, bh / img_h)

def pick_area_ratio():
    """
    Target area ratio distribution:
    - 70%: 1–6%
    - 20%: 6–12%
    - 10%: 12–25%
    """
    r = random.random()
    if r < 0.70:
        return random.uniform(0.01, 0.06)
    elif r < 0.90:
        return random.uniform(0.06, 0.12)
    else:
        return random.uniform(0.12, 0.25)


# -------------------------
# Main compositing
# -------------------------

def compose_one(bg_path, sprite_path, out_w=640, out_h=640, alpha_thresh=8):
    # Background (BGR)
    bg_full = cv2.imread(bg_path, cv2.IMREAD_COLOR)
    if bg_full is None:
        raise RuntimeError(f"Failed to read background: {bg_path}")

    bh_orig, bw_orig = bg_full.shape[:2]
    
    # Crop to square and resize to target size
    if bw_orig < out_w or bh_orig < out_h:
        bg = cv2.resize(bg_full, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    else:
        max_x = bw_orig - out_w
        max_y = bh_orig - out_h
        rand_x = random.randint(0, max_x)
        rand_y = random.randint(0, max_y)
        bg = bg_full[rand_y:rand_y + out_h, rand_x:rand_x + out_w]

    # Sprite (BGRA)
    sprite = read_sprite_rgba(sprite_path)

    # Crop sprite to its alpha bbox to remove empty margins
    bbox0 = alpha_bbox(sprite, alpha_thresh=alpha_thresh)
    if bbox0 is None:
        return None  # empty sprite
    sprite_c = crop_to_bbox(sprite, bbox0, pad=2)

    # Compute scale to achieve target area ratio in final image
    tgt_area_ratio = pick_area_ratio()

    # Use bbox area of the cropped sprite as an estimate of object area
    sh, sw = sprite_c.shape[:2]
    sprite_area = sw * sh
    target_area = tgt_area_ratio * (out_w * out_h)

    # scale so that (sw*scale)*(sh*scale) ≈ target_area
    scale = (target_area / max(1.0, sprite_area)) ** 0.5

    # Add a bit of randomness (±15%)
    scale *= random.uniform(0.85, 1.15)

    # Prevent crazy scales
    scale = max(0.02, min(scale, 2.0))

    sprite_r = resize_keep_aspect(sprite_c, scale)
    fh, fw = sprite_r.shape[:2]

    # Random placement, with chance to be partially off-screen (small probability)
    allow_cut = (random.random() < 0.10)  # 10% cutoffs
    if allow_cut:
        x = random.randint(-fw // 4, out_w - 1)
        y = random.randint(-fh // 4, out_h - 1)
    else:
        if fw >= out_w or fh >= out_h:
            # If too big, skip (or could downscale more)
            return None
        x = random.randint(0, out_w - fw)
        y = random.randint(0, out_h - fh)

    # Overlay
    bg2, _ = overlay_bgra_on_bgr(bg, sprite_r, x, y)
    if bg2 is None:
        return None

    # Recompute bbox in final image by rendering alpha mask onto blank mask
    # We'll build a mask by placing alpha into a blank canvas (same math as overlay clipping)
    mask = np.zeros((out_h, out_w), dtype=np.uint8)
    # clip coords
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(out_w, x + fw)
    y2 = min(out_h, y + fh)
    if x1 >= x2 or y1 >= y2:
        return None

    fx1 = x1 - x
    fy1 = y1 - y
    fx2 = fx1 + (x2 - x1)
    fy2 = fy1 + (y2 - y1)
    mask[y1:y2, x1:x2] = sprite_r[fy1:fy2, fx1:fx2, 3]

    ys, xs = np.where(mask > alpha_thresh)
    if len(xs) == 0:
        return None

    bx1, bx2 = int(xs.min()), int(xs.max())
    by1, by2 = int(ys.min()), int(ys.max())
    bbox = (bx1, by1, bx2, by2)

    # Convert to YOLO format
    yolo = yolo_from_bbox(bbox, out_w, out_h)
    return bg2, yolo


def main(
    sprites_dir="uav_ds/sprites",
    backgrounds_dir="uav_ds/backgrounds",
    out_dir="uav_ds/out",
    n=2000,
    out_w=640,
    out_h=640,
    seed=42
):
    random.seed(seed)
    np.random.seed(seed)

    sprites = list_images(sprites_dir, exts=("png",))
    bgs = list_images(backgrounds_dir, exts=("jpg", "jpeg", "png"))

    if not sprites:
        raise RuntimeError(f"No sprites found in: {sprites_dir}")
    if not bgs:
        raise RuntimeError(f"No backgrounds found in: {backgrounds_dir}")

    out_dir = Path(out_dir)
    img_out = out_dir / "images"
    lbl_out = out_dir / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    made = 0
    attempts = 0
    while made < n and attempts < n * 10:
        attempts += 1
        sp = random.choice(sprites)
        bg = random.choice(bgs)

        result = compose_one(bg, sp, out_w=out_w, out_h=out_h)
        if result is None:
            continue

        img, (cx, cy, bw, bh) = result

        # Save image
        stem = f"{made:06d}"
        img_path = img_out / f"{stem}.jpg"
        cv2.imwrite(str(img_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

        # Save YOLO label (single class id=0)
        lbl_path = lbl_out / f"{stem}.txt"
        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        made += 1
        if made % 200 == 0:
            print(f"Generated {made}/{n}")

    print(f"Done. Generated {made} images in {out_dir}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sprites", default="sprites")
    p.add_argument("--bgs", default="backgrounds")
    p.add_argument("--out", default="out")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--w", type=int, default=640)
    p.add_argument("--h", type=int, default=640)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    main(
        sprites_dir=args.sprites,
        backgrounds_dir=args.bgs,
        out_dir=args.out,
        n=args.n,
        out_w=args.w,
        out_h=args.h,
        seed=args.seed
    )