#!/usr/bin/env python3
"""
detection_script.py

Run from the Part_1_Glove_Detection folder.

Usage example:
python detection_script.py \
  --weights glove-yolov7/best.pt \
  --input input_images \
  --output output \
  --logs logs \
  --yolov7-dir yolov7 \
  --imgsz 640 \
  --conf 0.25 \
  --device cpu \
  --class-names "gloved_hand,bare_hand"
"""

import argparse
import os
import subprocess
import json
import cv2
from pathlib import Path

def run_yolov7_detect(yolov7_dir, weights, source, project, name, imgsz, conf, device):
    detect_py = os.path.join(yolov7_dir, "detect.py")
    if not os.path.exists(detect_py):
        raise FileNotFoundError(f"Could not find {detect_py}. Did you clone yolov7 repo into '{yolov7_dir}'?")
    cmd = [
        "python", detect_py,
        "--weights", weights,
        "--source", source,
        "--img-size", str(imgsz),
        "--conf-thres", str(conf),
        "--save-txt",
        "--save-conf",
        "--project", project,
        "--name", name,
        "--exist-ok"
    ]
    # add device option if provided and not 'cpu'
    if device and device.lower() != "cpu":
        cmd += ["--device", str(device)]
    print("Running YOLOv7 with command:\n", " ".join(cmd))
    subprocess.run(cmd, check=True)

def yolo_to_pixels(line, img_w, img_h):
    # YOLO line: cls x_center y_center width height [conf]
    vals = line.strip().split()
    if len(vals) < 5:
        return None
    cls = int(vals[0])
    x_c = float(vals[1])
    y_c = float(vals[2])
    w = float(vals[3])
    h = float(vals[4])
    conf = float(vals[5]) if len(vals) > 5 else None

    x_c_pix = x_c * img_w
    y_c_pix = y_c * img_h
    w_pix = w * img_w
    h_pix = h * img_h
    x1 = x_c_pix - w_pix / 2.0
    y1 = y_c_pix - h_pix / 2.0
    x2 = x_c_pix + w_pix / 2.0
    y2 = y_c_pix + h_pix / 2.0

    # Clip to image bounds and convert to ints
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w - 1, x2))
    y2 = max(0, min(img_h - 1, y2))

    return cls, conf, [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="Path to YOLOv7 weights .pt (e.g. glove-yolov7/best.pt)")
    p.add_argument("--input", required=True, help="Input folder with jpg/png images")
    p.add_argument("--output", default="output", help="Output folder where YOLOv7 writes annotated images")
    p.add_argument("--logs", default="logs", help="Folder where JSON logs will be saved")
    p.add_argument("--yolov7-dir", default="yolov7", help="Path to cloned yolov7 repo")
    p.add_argument("--imgsz", default=640, type=int, help="Image size for detection")
    p.add_argument("--conf", default=0.25, type=float, help="Confidence threshold")
    p.add_argument("--device", default="cpu", help="Device for detection: 'cpu' or '0'/'cuda:0'")
    p.add_argument("--class-names", default="gloved_hand,bare_hand",
                   help="Comma-separated class names in the same order as training (e.g. 'gloved_hand,bare_hand')")
    args = p.parse_args()

    weights = args.weights
    input_dir = args.input
    output_dir = args.output
    logs_dir = args.logs
    yolov7_dir = args.yolov7_dir
    imgsz = args.imgsz
    conf = args.conf
    device = args.device
    class_names = [s.strip() for s in args.class_names.split(",")]

    # Prepare directories
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(logs_dir).mkdir(parents=True, exist_ok=True)

    # 1) Run YOLOv7 detect on the whole folder (saves images + labels)
    run_yolov7_detect(yolov7_dir=yolov7_dir, weights=weights, source=input_dir,
                      project=output_dir, name="results", imgsz=imgsz, conf=conf, device=device)

    # 2) labels are stored in: <output_dir>/results/labels/*.txt
    labels_folder = os.path.join(output_dir, "results", "labels")
    annotated_images_folder = os.path.join(output_dir, "results")

    # 3) For each input image, build JSON log
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            input_path = os.path.join(root, fname)
            stem = Path(fname).stem
            label_file = os.path.join(labels_folder, stem + ".txt")
            img = cv2.imread(input_path)
            if img is None:
                print("Warning: could not read", input_path)
                continue
            h, w = img.shape[:2]

            detections = []
            if os.path.exists(label_file):
                with open(label_file, "r") as lf:
                    for line in lf:
                        parsed = yolo_to_pixels(line, w, h)
                        if parsed is None:
                            continue
                        cls, conf_score, bbox = parsed
                        label_name = class_names[cls] if cls < len(class_names) else str(cls)
                        detections.append({
                            "label": label_name,
                            "confidence": round(float(conf_score), 4) if conf_score is not None else None,
                            "bbox": bbox  # [x1,y1,x2,y2]
                        })

            # Save JSON
            json_log = {"filename": fname, "detections": detections}
            json_path = os.path.join(logs_dir, stem + ".json")
            with open(json_path, "w") as jf:
                json.dump(json_log, jf, indent=2)

            # (optionally) copy annotated image to logs or leave in output/results (YOLOv7 already saved it)
            annotated_path = os.path.join(annotated_images_folder, fname)
            if os.path.exists(annotated_path):
                # If you want annotated images in output root rather than output/results, you can copy/move
                pass

            print(f"Processed {fname}, {len(detections)} detections -> {json_path}")

    print("All done. Annotated images are in:", annotated_images_folder)
    print("JSON logs are in:", logs_dir)

if __name__ == "__main__":
    main()
