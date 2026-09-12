#!/usr/bin/env python3
"""量化 ONNX 模型，并用样本集核对掉点情况。

两种模式，**默认 int4**：

| 模式 | 做法 | 体积 | 排序一致性 ρ | ORT-web 加载 | ORT-web 推理 |
|---|---|---|---|---|---|
| `int8` | 动态量化：权重 INT8 + 逐通道 scale | 83.6 MB | 0.9983 | 0.43 s | 0.87 s |
| `int4` | 4-bit 仅权重（MatMulNBits，block=32） | **54.2 MB** | 0.9967 | 0.34 s | 1.90 s |

（体积与 ρ 是 Python 端实测；加载/推理是 ORT-web 1.29 的**单线程 WASM** 后端实测，
本机 GPU 是 SwiftShader，数字不代表真实硬件 —— 见 CLAUDE.md。）

选 int4 当默认的理由：**体积小 29 MB，而且精度反而更好**
（|ΔP| 中位数 0.0039 vs int8 的 0.0128）—— block-wise 每 32 个权重一个 scale，
比 int8 的逐通道一整个 scale 更贴合权重的局部结构。

代价是单线程 WASM 上推理慢一倍（要在读出时反量化）。但：
- 这个数字来自 SwiftShader 机器，**不可信**；真实硬件上（尤其 WebGPU）
  4-bit 的瓶颈在显存带宽，通常会反过来；
- 推理每张照片只跑一次，1.9 s 可以接受；而 29 MB 的下载量在首屏上是实打实的。

⚠️ 这个取舍**需要在真实硬件上复测**再定稿，别拿本机数字当结论。

⚠️ 关于体积目标：ViT-base 有 86M 参数，INT8 之后下限就是 ~86 MB，
**不可能压到 20-25 MB**。之前那个目标是把参数量估小了一个数量级得来的。
真要更小只能换更小的骨干（如 ViT-tiny 5.7M），那是换模型不是压模型。

用法：
    # 默认 int4
    tmp/venv-model/bin/python scripts/model/quantize_onnx.py --json tmp/quant_int4.json
    # 对比 int8
    tmp/venv-model/bin/python scripts/model/quantize_onnx.py --mode int8 \
        --out tmp/models/attractive.int8.onnx --json tmp/quant_int8.json

⚠️ 「准确率」在这一步要小心定义：这是二分类模型，但它的绝对概率有严重人群
偏移（见 CLAUDE.md），所以 **acc 看起来会很难看，且这个数字本身没有意义**。
真正要看的是量化前后的一致性 —— 同一张图的两个输出差多少、
排序有没有变。脚本两样都报，不要只看 acc。
"""

from __future__ import annotations

import argparse
import glob
import json
import os

MODEL_ID = "dima806/attractive_faces_celebs_detection"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="tmp/models/attractive.onnx")
    ap.add_argument("--out", default=None)
    ap.add_argument("--mode", default="int4", choices=["int8", "int4"])
    ap.add_argument("--block-size", type=int, default=32, help="int4 的分块大小")
    ap.add_argument("--sample-dir", default="tmp/faces/sample")
    ap.add_argument(
        "--weight-type",
        default="quint8",
        choices=["quint8", "qint8"],
        help="int8 模式：QUInt8 通常更准；QInt8 在部分后端更快",
    )
    ap.add_argument("--per-channel", action="store_true", default=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    import numpy as np
    import onnxruntime as ort
    from PIL import Image
    from transformers import AutoImageProcessor

    if not os.path.exists(args.input):
        raise SystemExit(f"找不到 {args.input}，先跑 export_onnx.py")

    if args.out is None:
        base, ext = os.path.splitext(args.input)
        args.out = f"{base}.{args.mode}{ext}"

    print(f"[1/3] 量化 {args.input}（模式 {args.mode}）")
    if args.mode == "int8":
        from onnxruntime.quantization import QuantType, quantize_dynamic

        wtype = QuantType.QUInt8 if args.weight_type == "quint8" else QuantType.QInt8
        print(f"      动态量化  权重 {args.weight_type}  逐通道 {args.per_channel}")
        quantize_dynamic(
            model_input=args.input,
            model_output=args.out,
            weight_type=wtype,
            per_channel=args.per_channel,
            # 不开启会多一次 ReduceRangeOfWeights，量化区间变小，长尾权重更容易被削平。
            # ViT 的注意力权重动态范围不小，这里保持默认的完整区间。
            reduce_range=False,
            extra_options={"MatMulConstBOnly": False},
        )
    else:
        from onnxruntime.quantization.matmul_4bits_quantizer import (
            DefaultWeightOnlyQuantConfig,
            MatMul4BitsQuantizer,
        )

        print(f"      4-bit 仅权重  block_size={args.block_size} 对称")
        # 只量化 MatMul 的权重（也就是 ViT 的 QKV / 投影 / MLP），
        # LayerNorm、patch embedding、分类头保持原样 —— 它们参数量很小，
        # 量化它们省不了多少，却容易把不变量搞坏。
        cfg = DefaultWeightOnlyQuantConfig(
            block_size=args.block_size,
            is_symmetric=True,
        )
        quantizer = MatMul4BitsQuantizer(
            model=args.input,
            algo_config=cfg,
        )
        quantizer.process()
        quantizer.model.save_model_to_file(args.out, use_external_data_format=False)

    fp32_mb = os.path.getsize(args.input) / 1024 / 1024
    q_mb = os.path.getsize(args.out) / 1024 / 1024
    print(f"      {fp32_mb:.1f} MB → {q_mb:.1f} MB（压缩 {fp32_mb / q_mb:.2f}×）")

    print("[2/3] 逐张比对量化前后")
    files = sorted(glob.glob(os.path.join(args.sample_dir, "*")))
    if not files:
        raise SystemExit("样本为空，先跑 scripts/calibration/fetch_sample.py")

    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    ref_sess = ort.InferenceSession(args.input, providers=["CPUExecutionProvider"])
    q_sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])

    def softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)

    rows = []
    for f in files:
        img = Image.open(f).convert("RGB")
        x = proc(images=img, return_tensors="np")["pixel_values"].astype(np.float32)
        ref = ref_sess.run(["logits"], {"pixel_values": x})[0][0]
        got = q_sess.run(["logits"], {"pixel_values": x})[0][0]
        rows.append({
            "file": os.path.basename(f),
            "ref_p": float(softmax(ref)[0]),
            "q_p": float(softmax(got)[0]),
            "max_logit_diff": float(np.abs(ref - got).max()),
            "argmax_same": bool(int(ref.argmax()) == int(got.argmax())),
        })

    d = np.array([r["max_logit_diff"] for r in rows])
    rp = np.array([r["ref_p"] for r in rows])
    qp = np.array([r["q_p"] for r in rows])
    same = sum(r["argmax_same"] for r in rows)

    print(f"      样本 {len(rows)} 张")
    print(f"      max|Δlogits|  中位 {np.median(d):.4f}  最大 {d.max():.4f}")
    print(f"      |ΔP(attr)|    中位 {np.median(np.abs(rp - qp)):.4f}  最大 {np.abs(rp - qp).max():.4f}")
    print(f"      argmax 一致   {same} / {len(rows)}")
    # Spearman 秩相关：量化把绝对分数挪一点没关系，**排序不能乱**。
    # 因为界面只用它做相对比较（见 CLAUDE.md「绝对概率不能当分数用」）。
    rank_ref = rp.argsort().argsort()
    rank_q = qp.argsort().argsort()
    rho = float(np.corrcoef(rank_ref, rank_q)[0, 1])
    print(f"      排序一致（Spearman ρ）{rho:.6f}   ← 这个才是关键指标")
    print("      注：不看 acc —— 这个模型的绝对概率有人群偏移，acc 本身没意义")

    print("[3/3] 完成")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({
                "mode": args.mode,
                "fp32_mb": fp32_mb, "quantized_mb": q_mb,
                "weight_type": args.weight_type if args.mode == "int8" else None,
                "block_size": args.block_size if args.mode == "int4" else None,
                "per_channel": args.per_channel if args.mode == "int8" else None,
                "median_logit_diff": float(np.median(d)), "max_logit_diff": float(d.max()),
                "argmax_same": int(same), "n": len(rows), "spearman": rho,
                "rows": rows,
            }, fh, ensure_ascii=False, indent=1)
        print(f"      已写入 {args.json}")


if __name__ == "__main__":
    main()
