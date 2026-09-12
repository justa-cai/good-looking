/**
 * 清晰度（模糊）检测。
 *
 * 为什么需要它：几何指标量的是关键点之间的**相对**位置，模糊不会让关键点整体
 * 偏移，但会让每个点的定位抖动变大 —— 分数照样给得出来，只是不可信。
 * 也就是说模糊是本项目里唯一「不会自然暴露自己」的坏输入，必须主动挡。
 *
 * 用的判据是**拉普拉斯方差**（variance of Laplacian），这是自动对焦领域
 * 常用的对焦度量：图像越清晰，二阶导数的响应越强、方差越大；越模糊，
 * 高频被抹掉，方差趋近于 0。
 *
 * ⚠️ 它本来是个「相对」量，绝对值受分辨率和对比度影响，所以这里做了归一：
 *   - **分辨率归一**：把人脸包围盒重采样到固定的 CROP_SIZE×CROP_SIZE 再算，
 *     这样瞳距 60px 和 300px 的照片可以直接比。
 *   - **对比度归一**：除以同一块区域的亮度标准差。否则一张高对比度的糊照片
 *     可能比一张低对比度的清晰照片方差还大。
 * 归一之后剩下的就是「细节密度」，阈值才有意义。
 */

import type { FaceLandmarks } from './types.ts'

/**
 * 归一化后的裁剪边长。取 160 是因为它明显大于拉普拉斯核的尺度（几个像素），
 * 又不至于把重采样本身的插值误差放大成细节。
 */
const CROP_SIZE = 160

/** 包围盒外扩比例。往里收会只剩五官中心，往外放会带进背景和头发纹理。 */
const PAD = 0.12

/** 3×3 拉普拉斯核（四邻域），对细线响应最灵敏。 */
const LAPLACIAN = [
  [0, 1, 0],
  [1, -4, 1],
  [0, 1, 0],
] as const

export interface SharpnessMeasurement {
  /** 归一化后的清晰度（拉普拉斯方差 / 亮度标准差），越大越清晰 */
  readonly sharpness: number
  /** 原始拉普拉斯方差，排查用 */
  readonly rawVariance: number
  /** 该区域的亮度标准差，排查用 */
  readonly lumaSd: number
}

/**
 * 量一张图里人脸区域的清晰度。
 *
 * 需要真正读像素，所以必须拿到 ImageBitmap（关键点本身不含像素信息）。
 * 任何一步失败（取不到 2D 上下文等）都返回 null —— 调用方按「测不了」处理，
 * 不要按「清晰」处理。
 */
export function measureSharpness(
  image: ImageBitmap,
  face: FaceLandmarks,
): SharpnessMeasurement | null {
  const box = faceBox(face)
  if (!box) return null

  const canvas = document.createElement('canvas')
  canvas.width = CROP_SIZE
  canvas.height = CROP_SIZE
  const ctx = canvas.getContext('2d', { willReadFrequently: true })
  if (!ctx) return null

  // 把（外扩后的）人脸框拉伸铺满 CROP_SIZE，顺带完成分辨率归一
  ctx.imageSmoothingEnabled = true
  ctx.imageSmoothingQuality = 'high'
  try {
    ctx.drawImage(
      image,
      box.x,
      box.y,
      box.w,
      box.h,
      0,
      0,
      CROP_SIZE,
      CROP_SIZE,
    )
  } catch {
    // 包围盒可能落在图外（关键点是推算出来的情况），drawImage 会抛
    return null
  }

  let data: Uint8ClampedArray
  try {
    data = ctx.getImageData(0, 0, CROP_SIZE, CROP_SIZE).data
  } catch {
    return null
  }

  // 转灰度并算均值
  const luma = new Float64Array(CROP_SIZE * CROP_SIZE)
  let sum = 0
  for (let i = 0, p = 0; i < luma.length; i++, p += 4) {
    // Rec.709 亮度权重
    const y = 0.2126 * data[p]! + 0.7152 * data[p + 1]! + 0.0722 * data[p + 2]!
    luma[i] = y
    sum += y
  }
  const mean = sum / luma.length

  let sdSum = 0
  for (const y of luma) sdSum += (y - mean) * (y - mean)
  const lumaSd = Math.sqrt(sdSum / luma.length)

  // 拉普拉斯卷积，只统计内部像素（边缘少一圈邻居）
  let lapSum = 0
  let lapSqSum = 0
  let count = 0
  for (let y = 1; y < CROP_SIZE - 1; y++) {
    for (let x = 1; x < CROP_SIZE - 1; x++) {
      let acc = 0
      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          const k = LAPLACIAN[ky + 1]![kx + 1]!
          if (k === 0) continue
          acc += k * luma[(y + ky) * CROP_SIZE + (x + kx)]!
        }
      }
      lapSum += acc
      lapSqSum += acc * acc
      count++
    }
  }
  const lapMean = lapSum / count
  const rawVariance = lapSqSum / count - lapMean * lapMean

  // 对比度归一：亮度完全均匀时（sd≈0）说明这块区域没有信息，
  // 归一化会除以 0，直接判为不可测。
  if (!Number.isFinite(lumaSd) || lumaSd < 1) return null

  return {
    sharpness: rawVariance / lumaSd,
    rawVariance,
    lumaSd,
  }
}

/** 关键点的包围盒，外扩 PAD 并夹到图像范围内。范围无效时返回 null。 */
function faceBox(face: FaceLandmarks) {
  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity
  for (const p of face.pixels) {
    if (p.x < minX) minX = p.x
    if (p.x > maxX) maxX = p.x
    if (p.y < minY) minY = p.y
    if (p.y > maxY) maxY = p.y
  }
  if (!Number.isFinite(minX) || !Number.isFinite(minY)) return null

  const w = maxX - minX
  const h = maxY - minY
  if (w <= 0 || h <= 0) return null

  const padX = w * PAD
  const padY = h * PAD
  const x = Math.max(0, minX - padX)
  const y = Math.max(0, minY - padY)
  const x2 = Math.min(face.width, maxX + padX)
  const y2 = Math.min(face.height, maxY + padY)
  if (x2 <= x || y2 <= y) return null

  return { x, y, w: x2 - x, h: y2 - y }
}
