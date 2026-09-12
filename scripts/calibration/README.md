# 几何指标的校准流程

`src/engine/calibration.ts` 里的数字**不是手填的**，是用 `src/engine/metrics.ts`
的测量式在一批真人照片上跑出来、再统计得到的。

改动任何测量式（哪怕只是换一个关键点、改一处投影方式）都必须重跑这个流程，
否则理想值会和实际测量对不上，分数整体偏移而没人发现 —— 这种错误不会报错，
只会让所有人看起来都偏丑或偏好看，非常难察觉。

## 依赖

需要 Chrome（CDP 可用）与能跑起 dev server 的环境。样本图片只在本机使用，
**不进仓库**（都是公有领域，但没必要把 37 张图塞进 git）。

## 步骤

```bash
# 1. 抓样本。默认抓到 tmp/faces/sample/
pnpm dev &                                     # 提供 src 模块
python3 scripts/calibration/serve.py 8099 tmp/faces &   # 提供图片（带 CORS）
python3 scripts/calibration/fetch_sample.py

# 2. 在浏览器里跑测量式，输出每张图的每个指标值
playwright-cli -s=cal eval "$(cat scripts/calibration/measure.js)" > tmp/calib_raw.txt

# 3. 统计出校准表
python3 scripts/calibration/stats.py tmp/calib_raw.txt
```

第 3 步会打印一张表：人群均值、总标准差、组内标准差、噪声占比、
以及判定出的理想值与来源（canon / empirical / typical）。
把它抄进 `calibration.ts`，并同步更新 `cal()` 的注释里引用的偏移量。

## 样本要求

- 正脸、单人、尽量少的遮挡；
- 需要**同一人多张**照片才能分离「人与人的差异」和「照片噪声」
  （`stats.py` 输出里的 `noise%` 就是这么来的）—— 没有重复样本时它算不出来；
- 目前的样本是 37 张美国国会官方肖像（公有领域），30 人，9 人有 2-3 张。
  这批人全是成年人、以中年为主、以欧美族裔为主 —— 换人群要重新评估。

## 判定规则

```
|实测均值 − 经典约定| ≤ 0.75σ  →  用经典值当理想值，标 'canon'
否则                          →  用实测均值，标 'empirical'（有经典约定）或 'typical'（本就没有）
```

`empirical` 意味着实测偏离经典约定超过 0.75σ。整批人同方向偏离，
只可能来自测量系统（关键点定义与解剖定义不同），不可能来自这批人本身。
所以要用实测值当理想值，并在界面上说清楚这是「接近人群平均」而非「符合公认比例」。

---

# 清晰度（模糊）阈值的校准

`src/face/quality.ts` 里的 `minSharpness` / `warnSharpness` 同样不是手填的。

因为删掉/新增关键点不影响这一项，触发重跑的条件不同：
**只有在改动 `src/face/sharpness.ts` 的度量方式（裁剪尺寸、外扩比例、归一化方式）
时才需要重跑。**

```bash
# 前置同上（dev server + serve.py），然后：
playwright-cli -s=cal eval "$(cat scripts/calibration/sharpness.js)" > tmp/sharpness_raw.txt
python3 scripts/calibration/sharpness_stats.py tmp/sharpness_raw.txt
```

脚本会把同一批「清晰」样本再合成出 `blur(1/2/3/4px)` 四档模糊样本，
打印每档的分布和一张「候选阈值 → 各档拒判率」的表，据此挑阈值。

两个必须遵守的前提（脚本头部注释里也有）：

1. **模糊必须用浏览器自己的 CSS 滤镜合成，不要用 PIL。**
   PIL 的 `GaussianBlur(radius)` 里 radius 就是 σ，和 CSS `blur(Npx)` 的标定差 3 倍以上
   （PIL σ=1.0 实测 10.05，CSS `blur(2px)` 实测约 2–4）。用错工具会把阈值调到错的地方。
2. **模糊要加在缩放之后的位图上。** 先糊原图再缩小，模糊半径会被缩放吃掉，
   严重低估糊的程度。

阈值选定后，同步更新 `quality.ts` 的注释和 CLAUDE.md 里的分布表。

