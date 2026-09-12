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
| ViT 模型许可证 | `apache-2.0` | HF API `cardData.license` |
| ViT 输入 | 224×224，mean/std 均 0.5，rescale 1/255，bicubic | `preprocessor_config.json` |
| ViT 标签 | 2 类（attractive / not attractive），报告准确率 83.8% | 模型卡 classification report |
| ViT 权重文件 | 根目录 `model.safetensors`（另有 checkpoint-149/3083 等中间检查点） | HF API `siblings` |

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
pnpm dev            # 本地开发（Vite dev server）
pnpm build          # 构建到 dist/
pnpm preview        # 预览构建产物（验证 base 路径是否正确）
```

部署到 GitHub Pages 时，Vite 的 `base` 必须是仓库名：
`base: '/good-looking/'`。`pnpm preview` 是验证这一点的最快方式——
构建后资源 404 基本就是 `base` 配错了。

## 验证清单

改完推理链路后，至少确认：

1. `pnpm build && pnpm preview` 能跑通，控制台无 404；
2. 上传一张正脸照能拿到分数，且**同一张图重复上传分数一致**（几何路线必须满足）；
3. 侧脸 / 模糊 / 无脸 的输入要有明确提示，不能静默给出错误分数。
