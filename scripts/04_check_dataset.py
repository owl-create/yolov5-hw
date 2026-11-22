"""第 4 步：数据集自检。

逐项核对验收最容易失分的几点：
  1) data/*.yaml 里的 nc 与 names 长度一致，且与标注文件里的类别下标一致；
  2) 标注框坐标都在 [0,1] 内、宽高为正，没有被裁坏；
  3) 验证集图片没有混进训练集（按文件内容 md5 比对，而不是只看文件名）；
  4) 统计各类标注数量、框尺寸分布，可直接抄进 README。

用法：
    python scripts/04_check_dataset.py --data data/hw.yaml --dataset dataset
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter

import yaml

IMG_EXT = (".jpg", ".jpeg", ".png")


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description="数据集自检")
    ap.add_argument("--data", default="data/hw.yaml")
    ap.add_argument("--dataset", default="dataset")
    args = ap.parse_args()

    problems: list[str] = []
    data = load_yaml(args.data)
    names = data.get("names", [])
    nc = data.get("nc")

    if not isinstance(names, list) or not names:
        problems.append("data/*.yaml 里没有可用的 names")
    if nc != len(names):
        problems.append(f"nc={nc} 与 names 长度 {len(names)} 不一致")

    report = {"classes": names, "splits": {}}

    for split in ("train", "val"):
        img_dir = os.path.join(args.dataset, "images", split)
        lbl_dir = os.path.join(args.dataset, "labels", split)
        if not os.path.isdir(img_dir):
            problems.append(f"缺少目录 {img_dir}")
            continue
        images = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXT))
        counts = Counter()
        widths, heights = [], []
        empty_labels = 0
        for img in images:
            stem = os.path.splitext(img)[0]
            lp = os.path.join(lbl_dir, stem + ".txt")
            if not os.path.exists(lp):
                problems.append(f"{split}: 图片 {img} 缺少标注文件")
                continue
            rows = [ln.strip() for ln in open(lp, encoding="utf-8") if ln.strip()]
            if not rows:
                empty_labels += 1
            for ln in rows:
                parts = ln.split()
                if len(parts) != 5:
                    problems.append(f"{split}/{stem}.txt 行格式错误: {ln}")
                    continue
                try:
                    cls = int(parts[0])
                    cx, cy, w, h = (float(v) for v in parts[1:])
                except ValueError:
                    problems.append(f"{split}/{stem}.txt 数值解析失败: {ln}")
                    continue
                if not 0 <= cls < len(names):
                    problems.append(f"{split}/{stem}.txt 类别下标 {cls} 越界（nc={len(names)}）")
                    continue
                if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)):
                    problems.append(f"{split}/{stem}.txt 坐标越界: {ln}")
                if w <= 0 or h <= 0:
                    problems.append(f"{split}/{stem}.txt 宽高非正: {ln}")
                counts[names[cls]] += 1
                widths.append(w)
                heights.append(h)
        report["splits"][split] = {
            "images": len(images),
            "empty_label_files": empty_labels,
            "boxes": {k: counts[k] for k in names},
            "median_box_wh": (
                [round(sorted(widths)[len(widths) // 2], 4), round(sorted(heights)[len(heights) // 2], 4)]
                if widths else None
            ),
        }

    # 跨 split 内容重复（验证集混入训练集的典型表现）：按文件内容 md5 比对，不看文件名
    hash_split: dict[str, str] = {}
    for split in ("train", "val"):
        img_dir = os.path.join(args.dataset, "images", split)
        if not os.path.isdir(img_dir):
            continue
        for f in sorted(os.listdir(img_dir)):
            if not f.lower().endswith(IMG_EXT):
                continue
            h = md5(os.path.join(img_dir, f))
            if h in hash_split and hash_split[h] != split:
                problems.append(f"{f} 与 {split} 中另一张图内容相同（{hash_split[h]} / {split} 重叠）")
            hash_split.setdefault(h, split)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if problems:
        print("\n发现问题：")
        for p in problems[:30]:
            print("  -", p)
        print(f"共 {len(problems)} 项")
        return 1
    print("\n自检通过：nc/names 一致、标注合法、验证集未混入训练集")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
