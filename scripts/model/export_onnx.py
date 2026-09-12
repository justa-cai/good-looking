#!/usr/bin/env python3
"""把路线 B 的 ViT 分类器导出成 ONNX，并逐项验证数值一致性。

模型：`dima806/attractive_faces_celebs_detection`（Apache-2.0）
- 结构 ViTForImageClassification，vit-base-patch16-224，224×224，12 层；
- **二分类**：id2label = {0: "attractive", 1: "not attractive"}，
  作者在 10384 张（5192/5192 平衡）上报告准确率 83.8%；
- 预处理：resize 224 (bilinear) → rescale 1/255 → normalize mean=std=0.5。

导出的是 **logits**，不含 softmax。理由：softmax 放到 JS 里做更灵活
（要不要温度、要不要做几何与模型的融合都在前端决定），而且对量化更友好。

用法：
    python3 scripts/model/export_onnx.py                    # 导出 + 验证
    python3 scripts/model/export_onnx.py --out tmp/models/attractive.onnx

⚠️ 导出完必须看验证结果。`max|Δlogits|` 应该在 1e-4 量级；
如果到了 1e-2 以上，说明算子或预处理对不上，别急着往下走量化 ——
量化的误差只会在这个基础上叠加，问题会被掩盖成「量化掉点」。

依赖：torch、transformers、onnx、onnxruntime、numpy、Pillow。

⚠️ **不要用系统 python**。本机全局环境的 torch / torchvision 是错配的
（`operator torchvision::nms does not exist`），换个干净的 venv：

    python3 -m venv tmp/venv-model
    tmp/venv-model/bin/pip install --index-url https://download.pytorch.org/whl/cpu \\
        "torch==2.4.0" "torchvision==0.19.0"
    tmp/venv-model/bin/pip install "transformers==4.46.3" "onnx==1.17.0" \\
        "onnxruntime==1.21.0" "numpy<2" pillow

venv 放在 tmp/ 下，不入库。之后所有 model 相关的脚本都用 `tmp/venv-model/bin/python` 跑。
"""

from __future__ import annotations

import argparse
import os
import sys

MODEL_ID = "dima806/attractive_faces_celebs_detection"

# 导出后参与数值比对的输入个数。随机输入能覆盖到 NaN / 形状错误，
# 但覆盖不到预处理的坑，所以还会额外过一张真实人脸图。
RANDOM_CASES = 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--out", default="tmp/models/attractive.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument(
        "--sample-image",
        default=None,
        help="用于真图比对的照片路径；给 sample/ 里的一张即可",
    )
    ap.add_argument(
        "--hf-home",
        default="tmp/hf",
        help="HF 下载缓存目录。默认放项目 tmp/ 下，不污染 ~/.cache",
    )
    args = ap.parse_args()

    # 必须在 import transformers 之前设，否则缓存路径已经定了
    os.environ.setdefault("HF_HOME", os.path.abspath(args.hf_home))

    import numpy as np
    import torch
    from transformers import AutoImageProcessor, ViTForImageClassification

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"[1/4] 加载模型 {args.model_id}")
    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = ViTForImageClassification.from_pretrained(args.model_id)
    model.eval()

    labels = {int(k): v for k, v in model.config.id2label.items()}
    print(f"      结构 {model.config.architectures}  输入 {model.config.image_size}px")
    print(f"      标签 {labels}")
    print(f"      size={processor.size} mean={processor.image_mean} std={processor.image_std}")

    # 包一层只返回 logits：ONNX 图里有个 dict 输出很麻烦，
    # 而且 ModelOutput 的字段名在不同 transformers 版本里变过。
    class LogitsOnly(torch.nn.Module):
        def __init__(self, inner: ViTForImageClassification) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            return self.inner(pixel_values=pixel_values, return_dict=False)[0]

    wrapped = LogitsOnly(model).eval()
    size = model.config.image_size

    print(f"[2/4] 导出 ONNX（opset {args.opset}）→ {out_path}")
    dummy = torch.zeros(1, 3, size, size, dtype=torch.float32)
    torch.onnx.export(
        wrapped,
        (dummy,),
        out_path,
        input_names=["pixel_values"],
        output_names=["logits"],
        # batch 维动态：将来可能一次算多张（比如把原图和镜像图一起喂进去）。
        # H/W 固定 —— 预处理里已经 resize 到 224，固定住能让量化更准。
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
        export_params=True,
    )
    print(f"      文件大小 {os.path.getsize(out_path) / 1024 / 1024:.1f} MB")

    print("[3/4] 逐项验证 PyTorch ↔ ONNX Runtime")
    import onnxruntime as ort

    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])

    def compare(name: str, pixel_values: np.ndarray) -> bool:
        with torch.no_grad():
            ref = wrapped(torch.from_numpy(pixel_values)).numpy()
        got = sess.run(["logits"], {"pixel_values": pixel_values})[0]
        diff = float(np.abs(ref - got).max())
        agree = int(ref.argmax(-1)[0]) == int(got.argmax(-1)[0])

        probs = np.exp(got - got.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        p_attr = [f"{p:.4f}" for p in probs[:, 0]]

        ok = diff < 1e-4 and agree
        print(
            f"      {'OK ' if ok else 'FAIL'} {name:<18} "
            f"max|Δlogits|={diff:.2e}  argmax一致={agree}  P(attractive)={p_attr[0]}"
        )
        return ok

    all_ok = True

    rng = np.random.default_rng(0)
    for i in range(RANDOM_CASES):
        x = rng.random((1, 3, size, size), dtype=np.float32)
        all_ok &= compare(f"随机输入 #{i}", x)

    if args.sample_image:
        from PIL import Image

        img = Image.open(args.sample_image).convert("RGB")
        print(f"      真图 {os.path.basename(args.sample_image)} 原始尺寸 {img.size}")
        x = processor(images=img, return_tensors="np")["pixel_values"].astype(np.float32)
        all_ok &= compare("真实人脸照", x)
    else:
        print("      跳过真图比对（没给 --sample-image）")

    print("[4/4] 完成" if all_ok else "[4/4] ⚠️ 有比对不通过，先别往下做量化")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
