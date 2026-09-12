#!/usr/bin/env python3
"""统计 sharpness.js 的输出，给出阈值选择建议。

用法：
    python3 scripts/calibration/sharpness_stats.py tmp/sharpness_raw.txt

输出：
    1. 每一档模糊程度的清晰度分布（最小/分位/最大）；
    2. 一串候选阈值，以及每个阈值下的拒判率 —— 据此挑 minSharpness；
    3. 清晰样本的最低几个值 —— 阈值不能贴着它们，要留余量。

怎么读结果：阈值选在「清晰样本拒判率 0%」和「模糊样本拒判率尽量高」
的折中点上，**并且要在清晰样本最小值下面留出余量** —— 校准样本是专业
肖像照，普通用户的照片整体会更软，贴着最小值卡会大量误伤。
"""

import json
import re
import sys


def load(path: str) -> list[dict]:
    text = open(path, encoding="utf-8").read()
    m = re.search(r"### Result\s*\n(.*?)\n### Ran", text, re.S)
    if not m:
        sys.exit("没找到 eval 结果，确认 playwright-cli 的 eval 输出完整送到了文件里")
    return json.loads(json.loads(m.group(1)))


def quantile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    return sorted_vals[min(len(sorted_vals) - 1, int(p * len(sorted_vals)))]


def main() -> None:
    rows = load(sys.argv[1] if len(sys.argv) > 1 else "tmp/sharpness_raw.txt")
    levels = sorted({int(k[1:]) for r in rows for k in r if re.fullmatch(r"b\d+", k)})

    print(f"样本 {len(rows)} 张\n")
    print("模糊半径 |    最小     p05     p25    中位     p75     p95     最大  | n")
    values = {}
    for r in levels:
        vals = sorted(v[f"b{r}"] for v in rows if v.get(f"b{r}") is not None)
        values[r] = vals
        if not vals:
            print(f"  {r}px    | （全部未通过质量校验，无数据）")
            continue
        print(
            f"  {r}px    | {vals[0]:7.2f} {quantile(vals, .05):7.2f} "
            f"{quantile(vals, .25):7.2f} {quantile(vals, .5):7.2f} "
            f"{quantile(vals, .75):7.2f} {quantile(vals, .95):7.2f} "
            f"{vals[-1]:7.2f}  | {len(vals)}"
        )

    base = values.get(min(levels), [])
    if not base:
        sys.exit("清晰对照组没有数据，阈值无从谈起")

    print("\n候选阈值下的拒判率（列是模糊半径）：")
    print("  阈值   " + "".join(f"{r:>8}px" for r in levels))
    # 只在有意义的区间里扫：低于清晰样本一半的值没意义，
    # 高于清晰样本中位数则连对照组的会拒掉一半，也不会有人选。
    lo_t = max(0.25, min(base) * 0.4)
    hi_t = quantile(base, 0.5)
    steps = 20
    for i in range(steps + 1):
        t = lo_t + (hi_t - lo_t) * i / steps
        cells = []
        for r in levels:
            vals = values.get(r) or []
            cells.append(
                "     n/a"
                if not vals
                else f"{100 * sum(1 for x in vals if x < t) / len(vals):7.1f}%"
            )
        print(f"  {t:5.2f}  " + "".join(cells))

    print(f"\n清晰样本最低 8 个：{[round(v, 2) for v in base[:8]]}")
    print("阈值要在最小值**下方**留余量，不要贴住它 —— 校准样本是专业肖像照。")


if __name__ == "__main__":
    main()
