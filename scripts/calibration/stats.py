#!/usr/bin/env python3
"""把 measure.js 的输出统计成校准表。

用法：
    python3 scripts/calibration/stats.py tmp/calib_raw.txt

输入是 playwright-cli 的原始输出（含 "### Result" 那一行），脚本会自己剥壳。

输出每人一组、每组多张时，会额外算出**组内标准差**（同一人不同照片的差异）。
组内方差占总方差的比例（noise%）就是该指标的噪声占比 —— 它决定了这个指标
可不可信，也决定了容差该不该放大。没有重复样本就算不出来，会给出 NaN。
"""

from __future__ import annotations

import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

# 有经典面部比例约定的指标：id -> (约定值, 说明)
# 值为 None 表示本就没有公认标准，只能用人群均值当理想值。
CANON: dict[str, tuple[float | None, str]] = {
    "upperOverMiddle": (1.0, "经典三等分：上庭 = 中庭"),
    "lowerOverMiddle": (1.0, "经典三等分：下庭 = 中庭"),
    "faceWidthOverHeight": (None, "本就没有公认标准"),
    "jawOverFaceWidth": (None, "本就没有公认标准"),
    "interCanthalOverIPD": (0.5, "两眼相距一只眼的宽度"),
    "noseOverInterCanthal": (1.0, "鼻宽 = 两内眼角间距"),
    "mouthOverNose": (1.5, "嘴宽 = 1.5 倍鼻宽"),
}

# 判定为 canon 的容差阈值（以标准差为单位）
CANON_TOLERANCE_SD = 0.75
# 容差 = 人群总标准差 × 该系数
TOLERANCE_FACTOR = 1.5

_DROP = {
    "official", "portrait", "congress", "116th", "117th", "the", "of", "us",
    "u.s", "cropped", "3x4", "4x5", "photo", "headshot", "sen", "rep", "hon",
    "alternate", "crop", "square", "slight", "2019",
}


def person_of(filename: str) -> str:
    """从文件名里猜出人名，用来把同一人的多张照片分到一组。

    样本文件名来自 Wikimedia，形如
    `12_1280px-Colin_Allred__official_portrait__117th_Congress__cropped_.jpg`。
    猜错只会让组内统计偏保守（把不同人当成同一人会高估噪声），不会算错总量。
    """
    s = filename.split("_", 1)[-1]
    s = re.sub(r"^\d+px-", "", s)
    s = re.sub(r"\.(jpg|jpeg|png)$", "", s, flags=re.I)
    s = re.sub(r"[()\[\]]", " ", s)
    tokens = [t for t in re.split(r"[\s,_]+", s) if t]
    tokens = [
        t for t in tokens
        if t.lower() not in _DROP and not re.fullmatch(r"\d+", t)
    ]
    return " ".join(tokens[:2]).lower()


def load(path: str) -> list[dict]:
    text = Path(path).read_text()
    # playwright-cli 的输出：一行 "### Result"，下一行是 JSON 字符串字面量
    m = re.search(r"### Result\n(.*)", text)
    if not m:
        raise SystemExit(f"在 {path} 里找不到 '### Result' 行；确认命令是否跑对")
    payload = m.group(1).strip()
    rows = json.loads(payload)
    if isinstance(rows, str):
        rows = json.loads(rows)
    return rows


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    rows = load(sys.argv[1])

    ok = [r for r in rows if "error" not in r]
    print(f"样本 {len(rows)} 张，成功 {len(ok)} 张\n")
    for r in rows:
        if "error" in r:
            print(f"  跳过: {r['file'][:50]}  ({r['error']})")

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in ok:
        groups[person_of(r["file"])].append(r["values"])
    repeated = sum(1 for v in groups.values() if len(v) > 1)
    print(f"共 {len(groups)} 人，其中 {repeated} 人有 2 张以上照片\n")

    print(f"{'指标':24} {'均值':>8} {'总sd':>8} {'组内sd':>8} {'噪声%':>7} "
          f"{'约定':>6} {'|Δ|/sd':>7} {'理想值':>8} {'容差':>8}  来源")
    print("-" * 110)

    out: dict[str, dict] = {}
    for mid, (canon, note) in CANON.items():
        per = [[v[mid] for v in vs if mid in v] for vs in groups.values()]
        per = [p for p in per if p]
        allv = [x for p in per for x in p]
        if not allv:
            print(f"{mid:24}   （样本里没有这个指标）")
            continue

        mean, sd = st.mean(allv), st.pstdev(allv)
        num = sum((len(p) - 1) * st.variance(p) for p in per if len(p) > 1)
        den = sum(len(p) - 1 for p in per if len(p) > 1)
        within = (num / den) ** 0.5 if den else float("nan")
        noise_share = (within ** 2) / (sd ** 2) if sd > 0 else float("nan")

        if canon is None:
            ideal, source = mean, "typical"
        elif abs(mean - canon) / sd <= CANON_TOLERANCE_SD:
            ideal, source = canon, "canon"
        else:
            ideal, source = mean, "empirical"

        tol = sd * TOLERANCE_FACTOR
        out[mid] = {
            "mean": round(mean, 4), "sd": round(sd, 4), "withinSd": round(within, 4),
            "noiseShare": round(noise_share, 4), "ideal": round(ideal, 4),
            "tolerance": round(tol, 4), "idealSource": source,
        }
        delta = f"{abs(mean - canon) / sd:7.2f}" if canon is not None else f"{'—':>7}"
        canon_s = f"{canon:6.2f}" if canon is not None else f"{'—':>6}"
        print(f"{mid:24} {mean:8.4f} {sd:8.4f} {within:8.4f} {noise_share * 100:6.1f}% "
              f"{canon_s} {delta} {ideal:8.4f} {tol:8.4f}  {source}   ({note})")

    print("\n把上面的 ideal / tolerance / 来源抄进 src/engine/calibration.ts，")
    print("并把 mean-sd-withinSd 一并填入 cal() 的前三个参数。\n")
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
