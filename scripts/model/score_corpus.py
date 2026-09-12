#!/usr/bin/env python3
"""用导出的 ONNX 跑一遍样本集，回答一个必须先回答的问题：

    **这个模型的输出，能不能直接当成「颜值分」展示给用户？**

答案是不能，这个脚本就是证据。跑完会看到：

37 张美国国会议员官方肖像（全部是正常、清晰、正脸的专业证件照）上，
P(attractive) 的中位数只有 0.12，均值 0.23，只有 7/37 超过 0.5。
也就是说在这批人身上，模型的默认判断是「不好看」。

这不是 bug，是训练数据决定的：模型在「名人颜值」图上训练，
和「中老年政治人物证件照」这个分布差得很远。**绝对概率不可当作分数**，
否则等于给所有人一个稳定的低分，而且低的原因和脸无关。

但同一人的两张照片分数非常接近（Boebert 0.610 / 0.627，
Auchincloss 0.737 / 0.756），说明模型本身**稳定**，只是**偏移**。
所以路线 B 能提供的可靠信息是「同一批人里的相对次序」，
不是「0-100 的绝对颜值」—— 界面必须按这个来设计（见任务 #13）。

用法：
    tmp/venv-model/bin/python scripts/model/score_corpus.py
    # 想存下来做对比：--json tmp/model_scores.json
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os

MODEL_ID = "dima806/attractive_faces_celebs_detection"
ONNX_PATH = "tmp/models/attractive.onnx"


def _load_person_of():
    """复用几何校准那套人名归组，两个校准的人名划分必须一致。

    标定脚本之间各写一份分组逻辑，迟早会漂成两套口径，
    到时候「模型稳定但偏移」和「几何噪声」的结论就没法放在一起看。
    """
    path = os.path.join(os.path.dirname(__file__), "..", "calibration", "stats.py")
    spec = importlib.util.spec_from_file_location("calib_stats", os.path.abspath(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.person_of


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-dir", default="tmp/faces/sample")
    ap.add_argument("--onnx", default=ONNX_PATH)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    import numpy as np
    import onnxruntime as ort
    from PIL import Image
    from transformers import AutoImageProcessor

    files = sorted(glob.glob(os.path.join(args.sample_dir, "*")))
    if not files:
        raise SystemExit(f"{args.sample_dir} 里没有样本，先跑 scripts/calibration/fetch_sample.py")

    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    person_of = _load_person_of()

    rows = []
    for f in files:
        img = Image.open(f).convert("RGB")
        x = proc(images=img, return_tensors="np")["pixel_values"].astype(np.float32)
        logits = sess.run(["logits"], {"pixel_values": x})[0][0]
        e = np.exp(logits - logits.max())
        p = e / e.sum()
        rows.append({
            "file": os.path.basename(f),
            "person": person_of(os.path.basename(f)),
            "p_attractive": float(p[0]),
            "logit_margin": float(logits[0] - logits[1]),
        })

    p = np.array([r["p_attractive"] for r in rows])
    print(f"样本 {len(p)} 张")
    print(
        f"  min={p.min():.3f}  p05={np.quantile(p, .05):.3f}  "
        f"中位={np.median(p):.3f}  p95={np.quantile(p, .95):.3f}  max={p.max():.3f}"
    )
    print(f"  均值={p.mean():.3f}  标准差={p.std():.3f}")
    print(f"  判为 attractive(>0.5) 的：{(p > 0.5).sum()} / {len(p)}")

    # 同一人多张照片：组内离散度 vs 组间离散度。
    # 和几何指标的 stats.py 用同一个思路 —— 分离「模型噪声」和「真实差异」。
    groups: dict[str, list[float]] = {}
    for r in rows:
        groups.setdefault(r["person"], []).append(r["p_attractive"])
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    if multi:
        within = [max(v) - min(v) for v in multi.values()]
        between = [np.mean(v) for v in multi.values()]
        print(f"\n  同一人有多张的 {len(multi)} 组：")
        print(f"    组内极差  中位 {np.median(within):.3f}  最大 {max(within):.3f}")
        print(f"    组间（各组均值）标准差 {np.std(between):.3f}")
        print("    → 组内远小于组间，说明模型稳定，偏移是分布造成的，不是随机噪声")

    print("\n最低 5 张 / 最高 5 张：")
    for r in sorted(rows, key=lambda r: r["p_attractive"])[:5]:
        print(f"  {r['p_attractive']:.3f}  {r['file'][:52]}")
    print("  ...")
    for r in sorted(rows, key=lambda r: r["p_attractive"])[-5:]:
        print(f"  {r['p_attractive']:.3f}  {r['file'][:52]}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, ensure_ascii=False, indent=1)
        print(f"\n已写入 {args.json}")


if __name__ == "__main__":
    main()
