# good-looking

浏览器端人脸颜值分析。上传照片 → 本地推理 → 出分，**照片永不上传服务器**。

## 硬约束

1. **纯前端推理**：所有推理在浏览器内完成，没有后端服务，没有任何上传接口。
2. **静态托管**：产物部署到 GitHub Pages，只能有静态文件。因此——
   - 不能用需要服务端的路由（用 hash 路由或单页）；
   - Vite `base` 必须是 `/<repo-name>/`，否则 GitHub Pages 子路径下资源全 404。
3. **可公开再分发**：仓库是公开的，任何入库的代码/权重/资产都必须许可证干净。

## 两条推理路线

### 路线 A：几何打分（MediaPipe FaceMesh）— 主力，零权重

`图片来源 → MediaPipe Face Landmarker → 478 个关键点 → 纯 TS 几何引擎 → 分项分 + 总分 → 可视化叠加`

- 模型与 WASM 全部来自 CDN，**仓库不含任何权重**。
- 输出是对**面部比例**的度量（三庭五眼、对称性、canthal tilt、midface ratio 等），
  不是「学出来的审美」。确定性：同图同分，可比对、可复现。
- 参考实现：`Blueturboguy07/freeharmony`（AGPL-3.0 —— **只读参考，不要拷代码进本仓库**，
  本仓库要保持许可证自由）。

### 路线 B：学习模型（ViT 二分类，Apache-2.0）

`图片来源 → 人脸检测 + 仿射对齐 → 224×224 → ViT ONNX (WebGPU/WASM) → 好看概率`

- 模型：`dima806/attractive_faces_celebs_detection`，**Apache-2.0**，base 是
  `google/vit-base-patch16-224-in21k`（同样 Apache-2.0）。
- 输出：`attractive` / `not attractive` 二分类 softmax，不是 1-5 连续分。
- **不对齐会显著影响结果**，必须走对齐。

## 已核实的事实（2026-09-12）

不要凭记忆改动这些，改之前先重新核实。

| 项 | 值 | 核实方式 |
|---|---|---|
| `@mediapipe/tasks-vision` 最新版 | `1.0.1` | `npm view @mediapipe/tasks-vision version` |
| `onnxruntime-web` 最新版 | `1.29.0` | `npm view onnxruntime-web version` |
| `vite` 最新版 | `8.3.0` | `npm view vite version` |
| `typescript` 最新版 | `7.0.2`（原生编译器） | `npm view typescript dist-tags` |
| 包管理器 | `pnpm` 10.12.1 | 本地 `pnpm -v`，CI 里同版本 |
| ViT 模型许可证 | `apache-2.0` | HF API `cardData.license` |
| ViT 输入 | 224×224，mean/std 均 0.5，rescale 1/255，bicubic | `preprocessor_config.json` |
| ViT 标签 | 2 类（attractive / not attractive），报告准确率 83.8% | 模型卡 classification report |
| ViT 权重文件 | 根目录 `model.safetensors`（另有 checkpoint-149/3083 等中间检查点） | HF API `siblings` |

### MediaPipe 资源 CDN（已核实可访问，返回 206）

- wasm 目录：`https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm/`
- 模型：`https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`（3.6 MB）

版本号写在 URL 里，跟着 npm 包的版本一起升，不要漏改。

### ⚠️ 本机浏览器的 GPU 性能数据不可信

开发机上的 Chrome（playwright / CDP 常驻）跑在 `--headless --disable-gpu --use-gl=disabled` 下，
WebGL renderer 实测是 **SwiftShader（纯软件光栅化）**，宿主那块 AMD Radeon HD 7850 没被用上。

所以：**在本机测出的任何「GPU 比 CPU 快/慢」的结论都不成立**，不要据此改代码。
需要 GPGPU 性能数据时必须换到有真实硬件加速的设备上测。

### 后端选择：运行时实测，不写死

`face/landmarker.ts` 默认 `delegate: 'auto'` —— 两个后端各建一次、各跑 3 帧计时，留下快的那个。

原因是没法预判：MediaPipe 的 GPU delegate 需要把结果从显存读回 CPU（`ReadPixels`），
在软件渲染下实测比 CPU **慢 4.7 倍**（217ms vs 46ms），但在真实硬件上又会反过来。
既然开发机测不出有效结论，就让它到用户的设备上自己量。代价是首次加载多约 1s。

实测（本机，SwiftShader）：
| 输入最长边 | 单帧耗时（CPU） |
|---|---|
| 1025px | 42 ms |
| 512px | 34 ms |
| 256px | 32 ms |

缩放对耗时影响很小（模型内部有自己的输入尺寸），所以 `MAX_DIM` 保持 1280，
换取更稳的关键点定位。

**给关键点做改动后务必验证两件事**：478 个点齐全；索引语义没串
（用 `tmp/` 里的脚本在真人照片上画出标注再肉眼核对，别只看数字）。

## 几何指标的校准（改测量式前必读）

`src/engine/calibration.ts` 里的均值/标准差/理想值**不是手填的**，是用
`src/engine/metrics.ts` 的测量式在一批真人照片上跑出来统计得到的。
流程见 `scripts/calibration/README.md`。

**改动任何测量式（换关键点、改投影方式、改归一化基准）都必须重跑校准。**
这类错误不会报错，只会让所有人看起来都偏丑或偏好看 —— 极难察觉。

当前校准样本：37 张美国国会官方肖像（公有领域），25-30 人，9 人有 2-3 张。
样本图片不入库（`tmp/faces/`，已 gitignore），用 `fetch_sample.py` 重新抓。

### 评分模型

```
分项分 = 100 · exp(−ln2 · (Δ/容差)²)      容差 = 人群标准差 × 1.5
综合分 = 各分组分的平均（分组等权，不是各指标等权）
```
偏离 0 得 100 分，偏离 = 容差 得 50 分。容差取 1.5σ 而不是 1σ，
否则一半的人每项都会低于 50 分，看起来像在惩罚普通人。

### 理想值有三种来源，界面上必须区分

| 来源 | 含义 | 判定 |
|---|---|---|
| `canon` | 符合公认面部比例 | 实测均值与经典约定差 ≤ 0.75σ |
| `empirical` | 接近人群平均（经典约定存在但实测系统性偏离） | 差 > 0.75σ |
| `typical` | 接近人群平均（本就没有公认标准） | — |

当前 7 个指标里，`upperOverMiddle`（三等分）与 `mouthOverNose`（嘴宽=1.5鼻宽）
的经典约定成立；其余 5 个是 `empirical`/`typical`。

`empirical` 的含义要说清楚：**整批人同方向偏离经典约定 2σ 以上，不可能来自人本身，
只能是测量系统的问题**（MediaPipe 的眼角点比解剖眼角靠内、鼻翼点比鼻翼外缘靠外）。
所以那些指标的分数是「像不像普通人」，不是「符不符合公认比例」。

### ⚠️ 单张照片的分数有 ±5~8 分的噪声

实测：同一个人不同照片的综合分极差在 **0.7 ~ 15.4 分**之间（9 人样本）。
差得多的那些（Wexton 49.1 vs 64.5）主要是姿态差异造成的。

所以：
- 不要把个位数的分差当成有意义的差异来解读或展示；
- 界面要明说这一点，别让用户拿两次分数对比；
- 这正是质量校验（拒判姿态过大的照片）必须做的原因。

### 已核实的引擎行为（2026-09-12）

- **确定性**：同一组关键点连算 5 次，综合分完全一致（浮点级）。
- **分数分布**（37 张样本）：均值 77.6，标准差 8.9，范围 49.1–93.5。
- 分项噪声占比（组内方差/总方差）：`faceWidthOverHeight` 12.6% 最可信，
  `lowerOverMiddle` 40.1% 最不可信 —— 后者在界面上要标出来。

## 头部姿态与输入质量

### 姿态只从变换矩阵取，不要自己拿关键点推

`src/face/pose.ts` 把 MediaPipe 的 4×4 变换矩阵（**行主序、行向量约定**，
平移在最后一行）分解成 yaw/pitch/roll。

试过用左右关键点的对称性当姿态判据，**噪声太大**：同一人不同照片的对称性差异
比不同人之间还大（关键点会随表情和光照漂移）。模型直接给的 3D 配准结果稳定得多。

roll 的符号用受控实验验证过：把图按 CSS `rotate(+θ)` 转，读数 ≈ +θ
（θ=10° 读数 8.6、θ=20° 读数 18.6、θ=-10° 读数 -11.5）。yaw/pitch 无法这样合成，
只验了量级合理（专业证件照都在 13° 以内）。

### ⚠️ 引擎不是「旋转不变」的 —— 转图会改变分数

引擎的测量式本身是旋转不变的（距离 + 沿中轴投影），但**实测转图会改变分数**：
同一张图转 0°/10°/20°/-10°，综合分是 80.1 / 81.2 / 76.7 / 70.0。

原因不在引擎，而在 MediaPipe：图一转，它给出的关键点位置就变了（模型不是旋转等变的，
加上重采样）。所以**姿态必须进质量校验**，不能因为「引擎设计上不受影响」就跳过。

依据：同一人不同照片的分数极差与姿态极差的相关系数 **0.91** —— 姿态是分数不稳的主因。

### 阈值及其依据（`src/face/quality.ts`）

| 判据 | 阈值 | 依据 |
|---|---|---|
| 姿态 | \|yaw/pitch/roll\| ≤ 20° | **安全网，不是精确刻度**。30 张专业证件照实测最大 9.8°/13.0°/8.8°，所以 20° 一定放过正常照；但样本里没有真正拍坏的照片，**找不到该拒判的临界点** |
| 瞳距下限 | ≥ 50 px | 逐步缩小照片重测：ipd ≥ 100px 时 \|Δ\| ≲ 2 分；50–70px 时 ≈ 4-5 分；≤ 35px 时 > 6 分，27px 到 13 分 |
| 瞳距警告 | < 100 px | 同上，偏差开始可见的位置 |
| 人脸出画 / 贴边 | 出画拒判，贴边（2%）警告 | 出画时边缘关键点是推算的，下巴一出画整个下庭就错了 |
| 清晰度下限 | ≥ 4.0（归一化拉普拉斯方差） | 见下方「模糊检测」 |
| 清晰度警告 | < 6.0 | 同上；约 8% 的清晰样本会落进这一档 |

**没有做闭眼检测**：实测眼睑开合度的个体差异极大（同一批人 0.17–0.46，差近 3 倍），
闭眼的和天生眼裂小的分不开，拿它拒判会大量误伤；而闭眼对几何比例几乎没影响。
不检查反而更正确。

### 模糊检测（`src/face/sharpness.ts`）

模糊是本项目里**唯一不会自然暴露自己的坏输入**：关键点之间的相对位置不变，
所以分数照样给得出来，只是每个点的定位抖动变大、分数不可信。
其它坏输入（无脸、侧脸、太小、被裁）都会在关键点上直接暴露，模糊不会。

判据是**拉普拉斯方差**（自动对焦领域的经典对焦度量），但做了两层归一化，
否则不同照片不可比：

- **分辨率归一**：把人脸包围盒（外扩 12%）重采样到固定 160×160 再算；
- **对比度归一**：除以同区域的亮度标准差。不然一张高对比度的糊照片方差可能更高。

归一之后剩下的量就是「细节密度」。实测分布（37 张官方肖像作清晰样本，
再对同一批图施加浏览器 CSS `blur(Npx)` 合成模糊样本）：

| 模糊半径 | 最小 | 中位 | 最大 |
|---|---|---|---|
| 0 px | 4.5 | 10.4 | 42.3 |
| 1 px | 0.6 | 5.5 | 13.3 |
| 2 px | 0.1 | 2.2 | 4.8 |
| 3 px | 0.1 | 1.1 | 2.5 |
| 4 px | 0.0 | 0.6 | 1.4 |

阈值取 **4.0 而不是分布交界处的 4.5**，是刻意留的余量：4.5 正好是这批
**专业肖像照**里最软的一张，普通用户随手拍的照片整体会更软，卡在 4.5 会误伤。
代价：`blur(2px)` 那档还剩约 11% 能过，`blur(3px)` 及以上 100% 拦下。

⚠️ **能力边界**：`blur(1px)` 这种轻微发虚和原本就清晰的照片**分不开**
（中位 5.5 vs 10.4，分布重叠严重）。这一关拦的是「明显糊」，不是「不够锐」。
不要试图调阈值去分这一档，分布本身不支撑。

⚠️ 测模糊**必须传位图**（`checkQuality(faces, bitmap)`）——要读原始像素。
省略位图时该检查被跳过，`report.sharpness === null`，表示「没测」，**不等于「清晰」**。
读不到像素（取不到 2D 上下文、包围盒落在图外）时同样返回 null 并跳过，不误判为清晰。

⚠️ 合成测试图必须用**浏览器自己的 CSS `blur(Npx)`**，不要用 PIL。
PIL 的 `GaussianBlur(radius)` 里 radius 就是 σ，和 CSS 的标定完全对不上：
PIL σ=1.0 实测 10.05，而 CSS `blur(2px)` 实测约 2–4 —— 差了 3 倍以上。
用错工具合成出来的样本会把阈值调到错的地方。

### 已核实：37 张正常证件照全部通过，异常输入全部正确拒判

异常用例在 `tmp/faces/quality_cases/`（临时，不入库）：无人脸、双人脸、
歪 30°、瞳距 36px、下巴被裁、`blur(2px)`、`blur(3px)`。

模糊用例由浏览器 CSS 滤镜在**缩放后的位图**上生成（`blur_css2.jpg` / `blur_css3.jpg`），
实测 3.93 / 1.94，都低于阈值 4.0 被拦下；对照组 `blur_css0.jpg` 实测 13.54 正常出分。

### 排查提示：图片取不到时的报错具有误导性

`createImageBitmap` 拿到 404 返回的 HTML 页面时报的是
`InvalidStateError: The source image could not be decoded`，
看起来像图片格式问题，其实是路径错了。**看到这个报错先查 HTTP 状态码**，
不要去找图片本身的毛病。这个坑踩过两次。

### ⚠️ GitHub Pages 没有 COOP/COEP，SharedArrayBuffer 不可用

这是本项目最重要的部署约束：**多线程 WASM 用不了**。

- GitHub Pages 不支持自定义响应头，所以拿不到 cross-origin isolation；
- 因此 `SharedArrayBuffer` 不存在，MediaPipe 和 ONNX Runtime 的**多线程后端都会退化为单线程**；
- 推论：dev server 也**故意不开** COOP/COEP（见 `vite.config.ts`）——保证本地和线上行为一致。
  否则会出现「本地飞快、线上慢十倍」这种只在部署后才暴露的问题；
- 后果：不要指望用 `numThreads` 调优；提速要靠 WebGPU（不依赖 SAB）和减小模型。


### ⚠️ 已排除的方案：SCUT-FBP5500

**不要再考虑这条路。** 原因（已核实）：

- `HCIILAB/SCUT-FBP5500-Database-Release` 仓库**没有任何 LICENSE 文件**
  （`LICENSE` / `LICENSE.md` / `LICENSE.txt` / `COPYING` 全部 404）→ 默认保留所有权利。
- 权重**不在仓库里**，托管在百度网盘（`pan.baidu.com/s/1OhyJsCMfAdeo8kIZd29yAw`，密码 `ateu`）
  和 Google Drive 上，无法在 CI 里自动获取。
- HF 上的镜像同样受限：`MnLgt/scut-fbp5500` 是 `license: other` 且**不含数据**（`usedStorage: 0`）。
- 仅限非商业研究用途，公开再分发权重有法律风险。

如果将来确实需要 1-5 连续分，正确做法是**开源训练代码、不发布权重**，让使用者自行获取数据集复现。

## 模型资产管理

权重**不入库**（`.gitignore` 已排除 `*.onnx` / `*.safetensors` / `*.pth`）。理由：仓库体积、
以及避免把许可证有疑问的二进制提交进公开仓库。

- 开发/构建期：模型放 `public/models/`（已 gitignore），本地 fetch 即可。
- 部署后：从外部 CDN 拉取 + `Cache API` 缓存，首次加载后离线可用。
- MediaPipe 的 wasm 与 `.task` 走 jsDelivr / Google 官方 CDN，不落仓库。

## 目录结构（约定）

```
good-looking/
├── index.html
├── src/
│   ├── main.ts              # UI 入口
│   ├── face/                # MediaPipe 封装：检测、关键点、对齐
│   ├── engine/              # 纯 TS 几何打分引擎，零依赖、无 DOM，必须可单测
│   └── model/               # onnxruntime-web 推理封装（路线 B）
├── scripts/                 # Python 导出 / 量化 pipeline（路线 B），产物写进 public/models/
├── public/                  # 静态资源；models/ 不入库
├── tmp/                     # 临时产物，不入库
└── .github/workflows/       # GitHub Actions 构建 + 部署 Pages
```

## 工程约定

- **`src/engine/` 必须保持零依赖、无 DOM**：纯函数，输入关键点、输出分数。
  这样它才可单测、可复用，也是路线 A 的核心资产。
- **不做简化版**：不要为了跑通而写 demo 版 / stub 版实现，残缺实现不如不写。
- **不建测试文件目录**，除非明确要求。
- **临时文件一律放 `./tmp/`**，不要写到系统 `/tmp`。
- 修改功能时不要顺手新建无关文件。

## 常用命令

```bash
pnpm dev            # 本地开发（Vite dev server，端口 5273）
pnpm build          # typecheck + 构建到 dist/
pnpm preview        # 预览构建产物（验证 base 路径是否正确）
pnpm typecheck      # 只跑类型检查
```

部署到 GitHub Pages 时，Vite 的 `base` 必须是仓库名（见 `vite.config.ts` 的 `REPO_NAME`）。
改了仓库名要同步改那里，CI 里有一道 base 路径校验会挡住不一致的情况
（这个错误在本地 dev 下看不出来，只有部署后才暴露）。

CI 用的 Action 版本（2026-09-12 核实）：`actions/checkout@v7`、`actions/setup-node@v7`、
`pnpm/action-setup@v6`、`actions/configure-pages@v6`、`actions/upload-pages-artifact@v5`、
`actions/deploy-pages@v5`。

## 验证清单

改完推理链路后，至少确认：

1. `pnpm build && pnpm preview` 能跑通，控制台无 404；
2. 上传一张正脸照能拿到分数，且**同一张图重复上传分数一致**（几何路线必须满足）；
3. 侧脸 / 模糊 / 无脸 的输入要有明确提示，不能静默给出错误分数。
