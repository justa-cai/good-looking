/**
 * 人脸对齐与裁剪 —— 路线 B 推理前的预处理。
 *
 * 为什么必须做：ViT 是在**裁剪好的人脸图**上训练的，直接喂整张照片
 * （带身体、背景、各种构图）等于换了个输入分布。而且照片一转，关键点会变，
 * 模型看到的画面也跟着变 —— 和几何路线遇到的「引擎不是旋转不变的」是同一个问题，
 * 只是模型对构图更敏感。
 *
 * 做法是**相似变换**（旋转 + 等比缩放 + 平移），不做透视校正：
 * - 用两个虹膜中心定「眼睛连线」，转成水平 → 解决滚转（roll）；
 * - 用瞳距当尺度基准 → 解决远近；
 * - 按固定比例把人脸摆到裁剪框里的固定位置 → 解决构图。
 *
 * ⚠️ 只校正 roll，**不校正 yaw/pitch**。那两者需要 3D 模型把脸「扳正」，
 * 而扳正会引入插值失真，也会凭空捏造侧脸看不到的那半边。
 * 侧脸本就不该出分，交给 quality.ts 的 pose 那关去拒判。
 */

import { LM } from './indices.ts'
import type { FaceLandmarks } from './types.ts'

/**
 * 裁剪框边长 = 瞳距 × 这个系数。
 *
 * 取 2.8 的依据：人脸（发际线到下巴）高度大约是瞳距的 1.4 倍，
 * 宽度（两颧）约 1.6 倍。2.8 倍瞳距的方框能把整个头装进去还留出四周余量，
 * 接近常见人脸数据集的构图。
 *
 * ⚠️ 这个系数**没有训练数据的标注可对照** —— 原作者没发布裁剪方式。
 * 2.8 是按人脸解剖比例推算的上限估计，不是拟合出来的。
 * 如果将来能拿到训练集的裁剪统计，应该用那个数字替换它。
 */
const CROP_IPD_RATIO = 2.8

/**
 * 眼睛连线在裁剪框里的高度位置（0 = 顶边，1 = 底边）。
 *
 * 取 0.42 而不是 0.5：额头通常比下巴短，把人脸摆正中人脸会整体偏上，
 * 所以眼睛线要略高于中线。同理下方留白多一点。
 */
const EYE_LINE_Y = 0.42

/** 输出尺寸。模型固定吃 224×224（导出时 H/W 就是静态的）。 */
export const ALIGN_SIZE = 224

export interface AlignedFace {
  /** 对齐后的位图，尺寸恒为 ALIGN_SIZE × ALIGN_SIZE */
  readonly bitmap: ImageBitmap
  /** 用到的几何信息，便于排查 */
  readonly ipd: number
  /** 滚转角（度），旋转量取的是它的相反数 */
  readonly rollDeg: number
}

/**
 * 把脸对齐并裁成正方形。
 *
 * 取不到虹膜点或几何退化（瞳距为 0）时返回 null —— 调用方按「没法对齐」处理，
 * 而不是拿一张没对齐的图硬跑。
 */
export async function alignFace(
  image: ImageBitmap,
  face: FaceLandmarks,
  size = ALIGN_SIZE,
): Promise<AlignedFace | null> {
  const irisA = face.pixels[LM.irisA]
  const irisB = face.pixels[LM.irisB]
  if (!irisA || !irisB) return null

  const dx = irisB.x - irisA.x
  const dy = irisB.y - irisA.y
  const ipd = Math.hypot(dx, dy)
  if (!Number.isFinite(ipd) || ipd < 1) return null

  // 眼睛连线相对水平的夹角。画布旋转用弧度，正值是顺时针。
  const rollRad = Math.atan2(dy, dx)
  const rollDeg = (rollRad * 180) / Math.PI

  const eyeMid = { x: (irisA.x + irisB.x) / 2, y: (irisA.y + irisB.y) / 2 }
  const cropSize = ipd * CROP_IPD_RATIO
  const scale = size / cropSize

  const canvas = new OffscreenCanvas(size, size)
  const ctx = canvas.getContext('2d')
  if (!ctx) return null

  ctx.imageSmoothingEnabled = true
  ctx.imageSmoothingQuality = 'high'

  // 先把「裁剪框中心」移到画布中心，再反向旋转把眼睛连线摆平，
  // 最后按 scale 放大。这样变换后的眼睛中点就落在 (size/2, EYE_LINE_Y*size)。
  ctx.setTransform(scale, 0, 0, scale, size / 2, size * EYE_LINE_Y)
  ctx.rotate(-rollRad)
  ctx.translate(-eyeMid.x, -eyeMid.y)
  ctx.drawImage(image, 0, 0)

  ctx.setTransform(1, 0, 0, 1, 0, 0)
  const bitmap = canvas.transferToImageBitmap()

  return { bitmap, ipd, rollDeg }
}
