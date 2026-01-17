import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def get_top_detection(result):
    """Return (xyxy, conf) for the highest-confidence detection or None."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None

    # boxes.conf is a tensor of shape (N,)
    confs = boxes.conf.detach().cpu().numpy()
    best_i = int(np.argmax(confs))

    xyxy = boxes.xyxy[best_i].detach().cpu().numpy()  # (4,)
    conf = float(confs[best_i])
    return xyxy, conf


def open_writer(output_path: str, fps: float, width: int, height: int):
    """Try a couple of codecs to make MP4 writing work across environments."""
    output_path = str(output_path)
    codecs = ["mp4v", "avc1"]  # mp4v usually works; avc1 sometimes works better

    for c in codecs:
        fourcc = cv2.VideoWriter_fourcc(*c)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if writer.isOpened():
            return writer, c

    raise RuntimeError("VideoWriter failed to open with known codecs (mp4v, avc1).")


def main():
    p = argparse.ArgumentParser(description="YOLO video inference with optional bbox video export.")
    p.add_argument("--weights", required=True, help="Path to .pt weights (e.g., best.pt)")
    p.add_argument("--source", required=True, help="Input video path")
    p.add_argument("--conf", type=float, default=0.10, help="Confidence threshold for YOLO")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--save", action="store_true", help="Save output video with drawn bbox")
    p.add_argument("--out", default="output.mp4", help="Output video path (used with --save)")
    p.add_argument("--max_frames", type=int, default=0, help="Process only first N frames (0 = all)")
    args = p.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Video not found: {source}")

    model = YOLO(args.weights)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save:
        writer, codec = open_writer(args.out, fps, width, height)
        print(f"✅ Writing output to: {args.out} (codec={codec}, {width}x{height}@{fps:.2f})")
    else:
        print(f"ℹ️ Save disabled. Processing {width}x{height}@{fps:.2f}")

    frame_count = 0
    frames_with_det = 0
    best_conf_overall = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Inference
        result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        top = get_top_detection(result)

        if top is not None:
            xyxy, conf = top
            frames_with_det += 1
            best_conf_overall = max(best_conf_overall, conf)

            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"drone {conf:.2f}",
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        if writer is not None:
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames... (detections in {frames_with_det})")

        if args.max_frames and frame_count >= args.max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()

    det_ratio = frames_with_det / max(1, frame_count)
    detected_any = frames_with_det > 0

    print("\n=== Summary ===")
    print(f"Frames processed:      {frame_count}")
    print(f"Frames with detection: {frames_with_det} ({det_ratio:.1%})")
    print(f"Best confidence:       {best_conf_overall:.3f}")
    print(f"Detected any drone:    {detected_any}")

    # Exit code can be useful in scripts/CI
    # 0 = detected something, 1 = detected nothing
    raise SystemExit(0 if detected_any else 1)


if __name__ == "__main__":
    main()