"""第 6 步：用训练好的权重对视频推理，输出 result.mp4 和 3 帧截图。

截图直接从 result.mp4 里读回来保存，保证和视频画面完全一致。

用法：
    python scripts/06_infer_video.py --weights best.pt --video assets/clip.mp4 \
        --out result.mp4 --frames-dir .
"""

from __future__ import annotations

import argparse
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

PALETTE = [
    (56, 156, 255), (80, 220, 120), (255, 128, 80), (200, 120, 255),
    (60, 220, 240), (255, 200, 60), (140, 140, 255), (90, 200, 200),
]


def draw(img, boxes, classes, names, colors):
    """boxes: Nx4 xyxy 像素坐标；classes: N；在图上画框并写“类别 置信度”。"""
    h, w = img.shape[:2]
    thick = max(2, int(round(min(h, w) / 320)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, min(h, w) / 1400)
    for (x1, y1, x2, y2), c in zip(boxes, classes):
        color = colors[int(c) % len(colors)]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thick, cv2.LINE_AA)
    for (x1, y1, x2, y2), c in zip(boxes, classes):
        color = colors[int(c) % len(colors)]
        label = names[int(c)]
        (tw, th), base = cv2.getTextSize(label, font, scale, 1)
        y0 = max(int(y1) - th - 6 - base, 0)
        cv2.rectangle(img, (int(x1), y0), (int(x1) + tw + 8, y0 + th + 6 + base), color, -1)
        cv2.putText(img, label, (int(x1) + 4, y0 + th + 3), font, scale, (20, 20, 20), 1, cv2.LINE_AA)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description="视频推理")
    ap.add_argument("--weights", default="best.pt")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="result.mp4")
    ap.add_argument("--frames-dir", default=".")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--start", type=float, default=0.0, help="起始秒")
    ap.add_argument("--duration", type=float, default=24.0, help="输出时长（秒），验收要求 10-30 秒")
    ap.add_argument("--out-fps", type=float, default=25.0)
    args = ap.parse_args()

    device = torch.device("cpu")
    model = DetectMultiBackend(args.weights, device=device, dnn=False, data=None, fp16=False)
    stride, names = model.stride, model.names
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"无法打开视频: {args.video}", file=sys.stderr)
        return 2
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2 * 2
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2 * 2
    # 输出帧率不能高于源帧率，否则视频会被“快放”
    out_fps = min(args.out_fps, src_fps)
    step = max(src_fps / out_fps, 1.0)
    max_frames = int(args.duration * out_fps)

    writer = None
    for fourcc in ("mp4v", "avc1", "H264"):
        wtr = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*fourcc), out_fps, (w, h))
        if wtr.isOpened():
            writer, used = wtr, fourcc
            break
        wtr.release()
    if writer is None:
        print("无法创建视频写入器（缺少 H.264/MPEG-4 编码支持）", file=sys.stderr)
        return 3
    print(f"写视频: {args.out} {w}x{h} @ {out_fps:.2f}fps, 编码 {used}, 目标 {args.duration}s", flush=True)

    idx = written = 0
    counts = Counter()
    per_frame: list[int] = []  # 每帧检测框数量，用于挑截图
    while written < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / src_fps
        idx += 1
        if t < args.start:
            continue
        if idx % int(round(step)) != 0:
            continue

        frame = cv2.resize(frame, (w, h)) if frame.shape[1] != w or frame.shape[0] != h else frame
        im = letterbox(frame, (args.imgsz, args.imgsz), stride=stride, auto=False)[0]
        im = np.ascontiguousarray(im.transpose((2, 0, 1))[::-1])
        tensor = torch.from_numpy(im).to(device).float()[None] / 255.0
        with torch.no_grad():
            pred = model(tensor, augment=False, visualize=False)
        det = non_max_suppression(pred, args.conf, args.iou, max_det=100)[0]

        if len(det):
            boxes = scale_boxes(tensor.shape[2:], det[:, :4].clone(), frame.shape).round().cpu().numpy()
            cls = det[:, 5].cpu().numpy().astype(int)
            for c in cls:
                counts[names[c]] += 1
            frame = draw(frame, boxes, cls, names, colors)

        writer.write(frame)
        per_frame.append(len(det))
        written += 1
        if written % 50 == 0:
            print(f"  {written}/{max_frames} 帧，当前累计检测框 {sum(counts.values())}", flush=True)

    cap.release()
    writer.release()

    # 从 result.mp4 里回读 3 帧：在视频的前/中/后三段里各挑“检测框最多”的一帧，
    # 这样截图一定是有目标的画面，且与视频画面完全一致
    cap = cv2.VideoCapture(args.out)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or written
    os.makedirs(args.frames_dir, exist_ok=True)
    third = max(total // 3, 1)
    for i in range(3):
        lo, hi = i * third, min((i + 1) * third, total)
        window = per_frame[lo:hi] or [0]
        best = lo + int(np.argmax(window)) if max(window) > 0 else min(lo + third // 2, total - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, best)
        ok, img = cap.read()
        if not ok:
            continue
        out_png = os.path.join(args.frames_dir, f"frame_{i + 1:02d}.png")
        cv2.imwrite(out_png, img)
        print(f"截图 {out_png} (第 {best} 帧, {per_frame[best] if best < len(per_frame) else 0} 个框)")
    cap.release()

    print(f"\n输出: {args.out}, 实际 {written / out_fps:.2f}s @ {out_fps:.2f}fps, {written} 帧")
    print("各类检测框数量:", dict(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
