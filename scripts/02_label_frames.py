"""第 2 步：用 COCO 预训练的 YOLOv5 给抽出的帧做初标（伪标注），再由人工抽查修正。

只保留目标类别，并把 COCO 的类别下标重映射成本项目的 0..N-1；
随后过滤掉过小的框（远小于目标的噪点框会让标注看起来“偏小”）。

用法：
    python scripts/02_label_frames.py --weights yolov5s.pt \
        --images dataset/frames --out dataset/labels_raw --classes person,car
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import cv2
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "yolov5"))

from models.common import DetectMultiBackend  # noqa: E402
from utils.augmentations import letterbox  # noqa: E402
from utils.general import non_max_suppression, scale_boxes  # noqa: E402

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def main() -> int:
    ap = argparse.ArgumentParser(description="COCO 预训练模型伪标注")
    ap.add_argument("--weights", default="yolov5s.pt")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", default="person", help="逗号分隔的 COCO 类名，顺序即项目里的类别下标")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--min-box-px", type=float, default=12.0, help="边长小于该像素数的框丢弃")
    ap.add_argument("--min-area-frac", type=float, default=0.0006, help="面积占比小于该值的框丢弃")
    args = ap.parse_args()

    target = [c.strip() for c in args.classes.split(",") if c.strip()]
    for c in target:
        if c not in COCO_NAMES:
            print(f"未知 COCO 类名: {c}", file=sys.stderr)
            return 2
    coco_ids = [COCO_NAMES.index(c) for c in target]
    remap = {cid: i for i, cid in enumerate(coco_ids)}

    device = torch.device("cpu")
    model = DetectMultiBackend(args.weights, device=device, dnn=False, data=None, fp16=False)
    stride, names = model.stride, model.names
    print(f"模型: {args.weights}, stride={stride}, 目标类别={dict(zip(target, coco_ids))}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    images = sorted(f for f in os.listdir(args.images) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    stats = Counter()
    confs: list[float] = []
    audit = {}

    for i, name in enumerate(images, 1):
        path = os.path.join(args.images, name)
        im0 = cv2.imread(path)
        if im0 is None:
            print(f"跳过无法读取的图片: {path}", file=sys.stderr)
            continue
        im = letterbox(im0, (args.imgsz, args.imgsz), stride=stride, auto=False)[0]
        im = im.transpose((2, 0, 1))[::-1]  # BGR->RGB, HWC->CHW
        im = np.ascontiguousarray(im)
        tensor = torch.from_numpy(im).to(device).float() / 255.0
        tensor = tensor[None]

        with torch.no_grad():
            pred = model(tensor, augment=False, visualize=False)
        det = non_max_suppression(pred, args.conf, args.iou, classes=coco_ids, max_det=300)[0]

        h0, w0 = im0.shape[:2]
        lines, kept = [], 0
        if len(det):
            det[:, :4] = scale_boxes(tensor.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in det.tolist():
                x1, y1, x2, y2 = xyxy
                bw, bh = max(x2 - x1, 0.0), max(y2 - y1, 0.0)
                if min(bw, bh) < args.min_box_px or (bw * bh) / (w0 * h0) < args.min_area_frac:
                    continue
                x1, y1 = max(x1, 0.0), max(y1, 0.0)
                x2, y2 = min(x2, float(w0)), min(y2, float(h0))
                cx, cy = (x1 + x2) / 2 / w0, (y1 + y2) / 2 / h0
                bw, bh = (x2 - x1) / w0, (y2 - y1) / h0
                if bw <= 0 or bh <= 0:
                    continue
                lines.append(f"{remap[int(cls)]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                kept += 1
                stats[target[remap[int(cls)]]] += 1
                confs.append(conf)

        stem = os.path.splitext(name)[0]
        with open(os.path.join(args.out, stem + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        audit[name] = kept
        if i % 25 == 0 or i == len(images):
            print(f"  {i}/{len(images)} 张，已保留 {sum(audit.values())} 个框", flush=True)

    summary = {
        "weights": os.path.abspath(args.weights),
        "classes": target,
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "images": len(images),
        "images_with_boxes": sum(1 for v in audit.values() if v),
        "boxes": dict(stats),
        "conf_mean": round(float(np.mean(confs)), 4) if confs else None,
        "conf_min": round(float(np.min(confs)), 4) if confs else None,
    }
    with open(os.path.join(args.out, "label_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
