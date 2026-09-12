#!/usr/bin/env python3
"""比对 JS 端（classifier_probe.js）与 Python 端（score_corpus.py）的模型输出。

**这个脚本的存在意义**：`src/model/classifier.ts` 里的归一化是手写的
（`x/127.5 - 1`），而 Python 端用的是官方 `AutoImageProcessor`。
两者只要差一点，模型在网页上和在标定脚本里就不是同一个东西 ——
而且这种错误不会报错，只会让所有分数悄悄偏移。

预期量级：两边缩放实现不同（canvas 双线性 vs PIL resample），
再加上 JS 用 int4 / Python 用 fp32，差值应该在 **0.05 以内**。
如果看到 0.1 以上，说明预处理写错了，不是量化误差。

用法：
    python3 scripts/model/verify_js_vs_py.py tmp/classifier_probe.json tmp/model_scores.json

退出码非 0 表示没通过。
"""

from __future__ import annotations

import json
import re
import statistics
import sys

# 允许的最大 |ΔP|。上面解释了为什么是这个量级。
MAX_TOLERANCE = 0.05
# 中位数超过这个就说明有系统性偏移（比如少乘/多乘了一个系数）
MAX_MEDIAN = 0.02


def load(path: str) -> dict:
    text = open(path, encoding="utf-8").read()
    # 允许直接把 playwright-cli 的原始输出丢进来
    m = re.search(r"### Result\s*\n(.*?)\n### Ran", text, re.S)
    if m:
        text = m.group(1)
    return json.loads(json.loads(text) if text.lstrip().startswith('"') else text)


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)

    js = load(sys.argv[1])
    py = load(sys.argv[2])

    print(f"JS 端：后端 {js.get('provider')}  加载 {js.get('loadMs')} ms")
    # 两个脚本的 JSON 外壳不同：score_corpus.py 直接写一个 list，
    # classifier_probe.js 包了一层（还带 provider / loadMs）。两种都收。
    py_rows = py["rows"] if isinstance(py, dict) and "rows" in py else py
    py_by = {r["file"]: r["p_attractive"] for r in py_rows if "file" in r}

    print(f"\n{'文件':44s} {'JS(int4)':>9} {'Py(fp32)':>9} {'|Δ|':>8}")
    deltas = []
    missing = []
    for r in js["rows"]:
        name = r["file"]
        ref = py_by.get(name)
        if ref is None:
            missing.append(name)
            continue
        d = abs(r["js_raw"] - ref)
        deltas.append(d)
        print(f"{name[:44]:44s} {r['js_raw']:9.4f} {ref:9.4f} {d:8.4f}")

    if missing:
        print(f"\n⚠️ {len(missing)} 个文件名对不上，检查两边样本目录是否一致")
    if not deltas:
        sys.exit("\n没有可比对的行")

    med = statistics.median(deltas)
    print(f"\n匹配 {len(deltas)} 张")
    print(f"  |ΔP| 中位 {med:.4f}   最大 {max(deltas):.4f}")
    print(f"  参考：int4 量化本身的 |ΔP| 中位 0.0039 / 最大 0.0314")

    ok = max(deltas) <= MAX_TOLERANCE and med <= MAX_MEDIAN
    if ok:
        print("\n✅ 通过：JS 预处理与 Python 端一致")
    else:
        print(
            f"\n❌ 不通过（阈值 中位≤{MAX_MEDIAN} 且 最大≤{MAX_TOLERANCE}）。\n"
            "   差值这么大不是量化误差能解释的，先查 classifier.ts 的归一化系数\n"
            "   和 INPUT_SIZE 是不是和 preprocessor_config.json 一致。"
        )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
