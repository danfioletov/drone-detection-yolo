# Drone Detection with YOLOv8 + Synthetic Shahed Dataset (Blender → YOLO)

This project explores drone detection using YOLOv8.  
It starts with a baseline model trained on public drone/bird datasets and then narrows down to **single-class Shahed-type UAV detection** using a **synthetic dataset pipeline** (Blender rendering + compositing + YOLO labels).

> Educational / research project. No operational use.

---

## Project Goals
- Train a baseline detector for flying objects (birds vs drones).
- Improve detection on UAVs using a drone detection dataset.
- Narrow the task to **single-class Shahed detection**.
- Build an end-to-end synthetic dataset pipeline:
  **3D model → rendered sprites → composited backgrounds → YOLO annotations → train → inference**.

---

## Datasets (external)
Baseline datasets used:
- Drone vs Bird (Kaggle): https://www.kaggle.com/datasets/muhammadsaoodsarwar/drone-vs-bird
- Drone Detection (Kaggle): https://www.kaggle.com/datasets/banderastepan/drone-detection

Synthetic Shahed dataset:
- 3D model: *(not included in this repository; see notes below)*
- Background images: *(not included in this repository)*

> This repo does not include datasets due to size/licensing.

---

## Repository Structure
- `scripts/generate_sprites_ds.py` — renders transparent PNG sprites (RGBA) from a 3D model in Blender.
- `scripts/compose_yolo.py` — composites sprites onto real backgrounds, generates YOLO bounding-box labels.
- `scripts/dataset_splitting_yolo.py` — splits dataset into train/val/test (70/20/10).
- `notebooks/train2.ipynb` — training in Google Colab (YOLOv8 fine-tuning).
- `scripts/inference-check.py` — local inference on video files and visualization.

---

## Setup
```bash
> pip install -r requirements.txt
```

## Synthetic Dataset Pipeline (Shahed)

1) Render sprites in Blender

Blender ≥ 4.x required for sprite generation.

Run Blender in background mode with the sprite generation script:
```bash
blender -b shahed2.blend -P scripts/generate_sprites_ds.py -- --out <SPRITES_DIR> --num 400 --res 640
```

## Compose sprites with backgrounds + create YOLO labels
```bash
python scripts/compose_yolo.py --sprites <SPRITES_DIR> --bgs <BACKGROUNDS_DIR> --out out --n 2000 --w 640 --h 640
```

## Split to train/val/test
```bash
python scripts/dataset_splitting_yolo.py
```

## Training (Google Colab)

Fine-tuning YOLOv8s on the synthetic single-class dataset.

Hyperparameters used:
- epochs=80
- batch=16
- patience=20

(Colab notebook: notebooks/train2.ipynb)

## Inference (Local)

python scripts/inference-check.ipynb

## Results (current)
- Baseline drone detector works on some UAV types but generalizes poorly to Shahed.
- Synthetic Shahed training is functional, but performance on real-world videos depends strongly on domain gap (motion blur, compression artifacts, lighting, scale distribution).

See assets/ for example outputs.

## Notes on 3D Model / Backgrounds

The 3D model and background images are not included in the repository.
To reproduce the synthetic dataset, download:
- a Shahed-type 3D model (OBJ/GLB/GLTF) compatible with Blender
- a set of background images (sky/clouds/landscape) for compositing
