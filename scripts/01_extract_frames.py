"""第 1 步：从视频中抽帧。

按固定帧率采样，再用 dHash 去掉几乎重复的画面，输出可直接标注的图片序列。

用法：
    python scripts/01_extract_frames.py --video assets/people-detection.mp4 --out dataset/frames --fps 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2


def dhash(gray, size: int = 8) -> int:
    """8x8 差分哈希，用于快速判断两帧是否几乎一样。"""
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = 0
    for v in diff.flatten():
        bits = (bits << 1) | int(v)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def main() -> int:
    ap = argparse.ArgumentParser(description="抽帧 + 去重")
    ap.add_argument("--video", required=True, help="输入视频路径")
    ap.add_argument("--out", required=True, help="输出图片目录")
    ap.add_argument("--fps", type=float, default=5.0, help="采样帧率（默认 5）")
    ap.add_argument("--start", type=float, default=0.0, help="起始秒")
    ap.add_argument("--end", type=float, default=None, help="结束秒（默认到结尾）")
    ap.add_argument("--max-side", type=int, default=0, help="最长边缩放到该值；0 表示保持原分辨率")
    ap.add_argument("--dedupe", type=int, default=4, help="dHash 汉明距离小于该值视为重复并跳过")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"无法打开视频: {args.video}", file=sys.stderr)
        return 2

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / src_fps if src_fps else 0.0
    end = duration if args.end is None else min(args.end, duration)

    os.makedirs(args.out, exist_ok=True)
    step = src_fps / args.fps  # 每多少原生帧取一帧
    idx, kept, last_hash = 0, 0, None
    scale = 1.0
    if args.max_side and max(width, height) > args.max_side:
        scale = args.max_side / max(width, height)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / src_fps
        idx += 1
        if t < args.start:
            continue
        if t > end:
            break
        if idx % max(int(round(step)), 1) != 0:
            continue

        if scale != 1.0:
            frame = cv2.resize(
                frame,
                (int(round(width * scale)), int(round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h = dhash(gray)
        if last_hash is not None and hamming(h, last_hash) < args.dedupe:
            continue
        last_hash = h
        kept += 1
        name = f"frame_{kept:05d}.jpg"
        cv2.imwrite(os.path.join(args.out, name), frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])

    cap.release()

    meta = {
        "video": os.path.abspath(args.video),
        "src_fps": round(src_fps, 3),
        "src_size": [width, height],
        "duration_s": round(duration, 3),
        "clip": [args.start, round(end, 3)],
        "sample_fps": args.fps,
        "frames_written": kept,
        "resize_scale": round(scale, 4),
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"视频 {width}x{height} @ {src_fps:.2f}fps, 时长 {duration:.2f}s -> 写出 {kept} 帧到 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
