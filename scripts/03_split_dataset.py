"""第 3 步：把标注好的帧划分成 train / val。

划分方式：先按帧序（时间顺序）排序，再按比例切成前后两段，并在切点处丢掉
一小段过渡帧，保证验证集画面不会和训练集“几乎同一时刻”。这样验证集里的
图片不会出现在训练集中，指标不会虚高。

用法：
    python scripts/03_split_dataset.py --frames dataset/frames --labels dataset/labels_raw \
        --out dataset --train-ratio 0.8
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

IMG_EXT = (".jpg", ".jpeg", ".png")


def main() -> int:
    ap = argparse.ArgumentParser(description="train/val 划分")
    ap.add_argument("--frames", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--gap-seconds", type=float, default=1.0, help="切点附近丢弃的秒数")
    ap.add_argument("--classes", required=True, help="逗号分隔，顺序与标注文件里的类别下标一致")
    args = ap.parse_args()

    names = [c.strip() for c in args.classes.split(",") if c.strip()]
    frames = sorted(f for f in os.listdir(args.frames) if f.lower().endswith(IMG_EXT))

    sample_fps = 5.0
    meta_path = os.path.join(args.frames, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            sample_fps = float(json.load(f).get("sample_fps", sample_fps))

    # 只保留有标注框的帧（纯背景帧不参与训练，避免空标签文件干扰统计）
    pairs = []
    skipped_empty = 0
    for img in frames:
        stem = os.path.splitext(img)[0]
        lp = os.path.join(args.labels, stem + ".txt")
        if not os.path.exists(lp):
            continue
        with open(lp, encoding="utf-8") as f:
            rows = [ln for ln in f.read().splitlines() if ln.strip()]
        if not rows:
            skipped_empty += 1
            continue
        pairs.append((img, stem, rows))

    n = len(pairs)
    if n == 0:
        print("没有找到带标注的帧，请先运行第 1、2 步", file=sys.stderr)
        return 2

    gap = max(1, int(round(args.gap_seconds * sample_fps)))
    n_train = int(round(n * args.train_ratio))
    train = pairs[:n_train]
    val = pairs[n_train + gap:]
    dropped_gap = pairs[n_train:n_train + gap]

    for split in ("train", "val"):
        os.makedirs(os.path.join(args.out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.out, "labels", split), exist_ok=True)
        for f in os.listdir(os.path.join(args.out, "images", split)):
            os.remove(os.path.join(args.out, "images", split, f))
        for f in os.listdir(os.path.join(args.out, "labels", split)):
            os.remove(os.path.join(args.out, "labels", split, f))

    per_split = {}
    for split, items in (("train", train), ("val", val)):
        counts = {k: 0 for k in names}
        for img, stem, rows in items:
            shutil.copy2(os.path.join(args.frames, img), os.path.join(args.out, "images", split, img))
            with open(os.path.join(args.out, "labels", split, stem + ".txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(rows) + "\n")
            for r in rows:
                counts[names[int(r.split()[0])]] += 1
        per_split[split] = {"images": len(items), "boxes": counts}

    stats = {
        "classes": names,
        "train_ratio": args.train_ratio,
        "sample_fps": sample_fps,
        "gap_frames_dropped": len(dropped_gap),
        "frames_without_boxes_skipped": skipped_empty,
        "splits": per_split,
        "total_images": len(train) + len(val),
        "total_boxes": {k: per_split["train"]["boxes"][k] + per_split["val"]["boxes"][k] for k in names},
    }
    with open(os.path.join(args.out, "dataset_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
