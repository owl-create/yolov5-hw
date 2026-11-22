"""第 5 步：训练入口。

把实际使用的超参数写死在脚本里，仓库复现时直接跑这一条命令即可得到同一套设置。

用法：
    python scripts/05_train.py --data data/hw.yaml --weights yolov5s.pt
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser(description="训练 YOLOv5（固定超参与随机种子）")
    ap.add_argument("--data", default="data/hw.yaml")
    ap.add_argument("--weights", default="yolov5s.pt", help="预训练权重；填 '' 表示从零训练")
    ap.add_argument("--img", type=int, default=416)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="hw-yolov5s")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--cache", default="ram", help="ram/disk/''")
    ap.add_argument("--extra", default="", help="附加参数，原样追加")
    args = ap.parse_args()

    cmd = [
        sys.executable, os.path.join(REPO, "yolov5", "train.py"),
        "--data", args.data,
        "--img", str(args.img),
        "--epochs", str(args.epochs),
        "--batch", str(args.batch),
        "--workers", str(args.workers),
        "--device", args.device,
        "--seed", str(args.seed),
        "--project", os.path.join(REPO, "runs", "train"),
        "--name", args.name,
        "--exist-ok",
        "--patience", str(args.patience),
    ]
    if args.weights:
        cmd += ["--weights", args.weights]
    if args.cache:
        cmd += ["--cache", args.cache]
    if args.extra:
        cmd += args.extra.split()

    print("运行:", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=REPO)

    csv_path = os.path.join(REPO, "runs", "train", args.name, "results.csv")
    if rc == 0 and os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows:
            last = {k.strip(): v for k, v in rows[-1].items()}
            keys = [k for k in last if k in ("epoch", "metrics/precision", "metrics/recall",
                                             "metrics/mAP_0.5", "metrics/mAP_0.5:0.95")]
            print("\n最后一轮验证集指标:")
            for k in keys:
                print(f"  {k}: {last[k]}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
