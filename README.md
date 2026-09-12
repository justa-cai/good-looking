# good-looking

浏览器端人脸颜值分析。上传照片 → 本地推理 → 出分。**照片永远不离开你的设备**。

纯静态前端，无后端，部署在 GitHub Pages 上。

## 状态

早期开发中，尚未可用。

## 计划中的能力

分析在浏览器内完成，分两条互补的链路：

- **面部比例度量** — 用 MediaPipe FaceMesh 的 478 个关键点计算三庭五眼、面部对称性、
  canthal tilt、midface ratio 等几何指标。确定性计算，同一张图永远得到同一个分数。
- **学习模型评分** — 基于 ViT 的二分类模型（Apache-2.0），输出「好看概率」。
  需要人脸对齐后送入推理。

## 隐私

这是本项目的核心约束，不是可选特性：

- 照片不上传，全部推理在浏览器内（WebGPU / WASM）完成；
- 没有后端服务，没有上传接口；
- 模型在首次加载后由浏览器缓存，之后可离线使用。

## 开发

```bash
pnpm install
pnpm dev      # 本地开发
pnpm build    # 构建
pnpm preview  # 预览构建产物
```

## 致谢与许可

- 人脸关键点检测：[MediaPipe Tasks Vision](https://github.com/google-ai-edge/mediapipe)（Apache-2.0）
- 学习模型：[dima806/attractive_faces_celebs_detection](https://huggingface.co/dima806/attractive_faces_celebs_detection)（Apache-2.0），
  base 为 [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k)
- 几何指标的美学约定参考公开的面部美学文献

## 免责声明

分数是基于照片的估算，用于娱乐和自我参考，**不是临床测量**。光照、镜头距离、拍摄角度
都会显著影响结果。请勿将其用于医疗、招聘、筛选或其他对人的评判性决策。
