/**
 * 结果图上的标注绘制。
 *
 * 只用 Canvas 2D，不引第三方绘图库：要画的东西很少（点、线、文字），
 * 引一个图形库不划算，也容易和推理用的 canvas 抢上下文。
 *
 * 两条路径共用这一份：静态照片传 ImageBitmap，实时模式传 `<video>`。
 * 两者都是 `CanvasImageSource`，`drawImage` 一视同仁，所以不需要分叉。
 */

import { LM } from '../face/indices.ts'
import type { FaceLandmarks, Point3 } from '../face/types.ts'
import type { AnalysisResult } from '../engine/index.ts'
import type { MetricId } from '../engine/metrics.ts'

/** 分项分 → 颜色。低分红、中黄、高分绿。 */
export function scoreColor(score: number): string {
  if (!Number.isFinite(score)) return '#6c7488'
  const t = Math.max(0, Math.min(1, score / 100))
  // 0 → 红(0°)，50 → 黄(≈60°)，100 → 绿(145°)
  return `hsl(${t * 145} 68% ${56 + t * 10}%)`
}

/** 要画的测量段：直接连两个关键点。 */
const SEGMENTS: ReadonlyArray<{ metric: MetricId; from: number; to: number }> = [
  { metric: 'faceWidthOverHeight', from: LM.contourA, to: LM.contourB },
  { metric: 'jawOverFaceWidth', from: LM.jawA, to: LM.jawB },
  { metric: 'noseOverInterCanthal', from: LM.alaA, to: LM.alaB },
  { metric: 'mouthOverNose', from: LM.mouthCornerA, to: LM.mouthCornerB },
]

/** 纵向比例的分界高度（沿中轴的投影位置），与引擎的测量方式一致。 */
const AXIAL_LEVELS: ReadonlyArray<number> = [LM.foreheadTop, LM.nasion, LM.subnasale, LM.chin]

export interface OverlayOptions {
  /** 画布的最大 CSS 宽度 */
  maxWidth?: number
  /** 是否把 478 个关键点都画出来 */
  showAllPoints?: boolean
  /**
   * 用这些点作图，而不是 `face.pixels`。
   *
   * 实时模式传的是 EMA 平滑过的点（原始点每帧都在抖，看起来像检测不稳），
   * 但**分数用的始终是未平滑的那一帧** —— 叠加层是取景辅助，不是测量结果。
   */
  points?: readonly Point3[]
}

/**
 * 按底图尺寸设好画布的像素尺寸与 CSS 尺寸。
 *
 * **尺寸没变就直接返回，一个字节都不动。** 给 `canvas.width` 重新赋值（哪怕赋的
 * 是同一个数）会重新分配后备存储并丢掉合成图层 —— 在 dpr=2、宽 620 时那是约
 * 3.5MB，实时模式每秒做十几次是纯粹的浪费。所以判定放在这个函数里，
 * `drawOverlay` 每帧无脑调它就行。
 *
 * @returns 是否真的重设了
 */
export function syncCanvasSize(
  canvas: HTMLCanvasElement,
  srcW: number,
  srcH: number,
  maxWidth = 620,
): boolean {
  const cssW = Math.min(maxWidth, srcW)
  const dpr = Math.min(window.devicePixelRatio || 1, 2)
  const w = Math.round(cssW * dpr)
  const h = Math.round((srcH / srcW) * cssW * dpr)
  const styleW = `${cssW}px`

  if (canvas.width === w && canvas.height === h && canvas.style.width === styleW) return false

  canvas.width = w
  canvas.height = h
  canvas.style.width = styleW
  canvas.style.maxWidth = '100%'
  return true
}

/**
 * 把底图和标注一起画到 canvas。
 * canvas 的像素尺寸按设备像素比放大，保证高分屏上不糊。
 *
 * @param result 分项分。实时模式在样本还没攒够时拿不到，传 null ——
 *   这时测量段一律画成灰色（`scoreColor(NaN)`），而几何标注照常有用。
 */
export function drawOverlay(
  canvas: HTMLCanvasElement,
  image: CanvasImageSource,
  face: FaceLandmarks,
  result: AnalysisResult | null,
  opts: OverlayOptions = {},
): void {
  const { maxWidth = 620, showAllPoints = true } = opts
  const pts = opts.points ?? face.pixels
  const srcW = face.width
  const srcH = face.height

  // 尺寸没变时这是个空操作，见 syncCanvasSize
  syncCanvasSize(canvas, srcW, srcH, maxWidth)

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const cssW = Math.min(maxWidth, srcW)
  const dpr = Math.min(window.devicePixelRatio || 1, 2)

  // 之后一律在「原图坐标系」里作图，缩放交给变换矩阵
  ctx.setTransform(dpr * (cssW / srcW), 0, 0, dpr * (cssW / srcW), 0, 0)
  ctx.clearRect(0, 0, srcW, srcH)
  ctx.drawImage(image, 0, 0, srcW, srcH)

  const top = pts[LM.foreheadTop]
  const chin = pts[LM.chin]
  if (!top || !chin) return

  const axisAngle = Math.atan2(chin.y - top.y, chin.x - top.x)
  // 垂直于中轴的方向（单位向量）—— 三庭分界线沿这个方向画
  const nx = -Math.sin(axisAngle)
  const ny = Math.cos(axisAngle)

  const scoreOf = (id: MetricId): number =>
    result?.metrics.find((m) => m.id === id)?.score ?? Number.NaN

  // ---- 478 个关键点：很淡，作为「确实检测到了」的视觉证据 ----
  if (showAllPoints) {
    ctx.fillStyle = 'rgba(110, 168, 254, 0.3)'
    const dot = Math.max(0.8, srcW / 1000)
    for (const p of pts) {
      ctx.beginPath()
      ctx.arc(p.x, p.y, dot, 0, Math.PI * 2)
      ctx.fill()
    }
  }

  const ca = pts[LM.contourA]
  const cb = pts[LM.contourB]
  const halfWidth = ca && cb ? Math.hypot(ca.x - cb.x, ca.y - cb.y) / 2 : srcW * 0.15

  // ---- 中轴 ----
  ctx.save()
  ctx.strokeStyle = 'rgba(255,255,255,0.42)'
  ctx.lineWidth = 1
  ctx.setLineDash([5, 5])
  ctx.beginPath()
  ctx.moveTo(top.x, top.y)
  ctx.lineTo(chin.x, chin.y)
  ctx.stroke()
  ctx.restore()

  // ---- 三庭分界：过关键点、垂直于中轴 ----
  // 上段与中段由 upperOverMiddle 决定，中段与下段由 lowerOverMiddle 决定，
  // 所以四条线的颜色是「上上、下下」而不是每段一色。
  const upperColor = scoreColor(scoreOf('upperOverMiddle'))
  const lowerColor = scoreColor(scoreOf('lowerOverMiddle'))
  const levelColors = [upperColor, upperColor, lowerColor, lowerColor]

  AXIAL_LEVELS.forEach((index, k) => {
    const p = pts[index]
    if (!p) return
    ctx.save()
    ctx.strokeStyle = levelColors[k]!
    ctx.lineWidth = 1.6
    ctx.setLineDash([7, 5])
    ctx.beginPath()
    ctx.moveTo(p.x - nx * halfWidth * 1.05, p.y - ny * halfWidth * 1.05)
    ctx.lineTo(p.x + nx * halfWidth * 1.05, p.y + ny * halfWidth * 1.05)
    ctx.stroke()
    ctx.restore()
  })

  // ---- 横向测量段 ----
  for (const seg of SEGMENTS) {
    const a = pts[seg.from]
    const b = pts[seg.to]
    if (!a || !b) continue
    const color = scoreColor(scoreOf(seg.metric))
    ctx.save()
    ctx.strokeStyle = color
    ctx.lineWidth = 1.8
    ctx.beginPath()
    ctx.moveTo(a.x, a.y)
    ctx.lineTo(b.x, b.y)
    ctx.stroke()
    for (const p of [a, b]) {
      ctx.beginPath()
      ctx.arc(p.x, p.y, Math.max(2, srcW / 420), 0, Math.PI * 2)
      ctx.fillStyle = color
      ctx.fill()
      ctx.strokeStyle = 'rgba(0,0,0,0.55)'
      ctx.lineWidth = 1
      ctx.stroke()
    }
    ctx.restore()
  }

  // ---- 眼距：内眼角间距 + 瞳距基准（所有长度的单位）----
  const innerA = pts[LM.eyeInnerA]
  const innerB = pts[LM.eyeInnerB]
  if (innerA && innerB) {
    ctx.save()
    ctx.strokeStyle = scoreColor(scoreOf('interCanthalOverIPD'))
    ctx.lineWidth = 1.8
    ctx.beginPath()
    ctx.moveTo(innerA.x, innerA.y)
    ctx.lineTo(innerB.x, innerB.y)
    ctx.stroke()
    ctx.restore()
  }

  const irisA = pts[LM.irisA]
  const irisB = pts[LM.irisB]
  if (irisA && irisB) {
    ctx.save()
    ctx.strokeStyle = 'rgba(255,255,255,0.9)'
    ctx.lineWidth = 1.3
    ctx.setLineDash([4, 4])
    ctx.beginPath()
    ctx.moveTo(irisA.x, irisA.y)
    ctx.lineTo(irisB.x, irisB.y)
    ctx.stroke()
    ctx.restore()
    for (const p of [irisA, irisB]) {
      ctx.beginPath()
      ctx.arc(p.x, p.y, Math.max(2.4, srcW / 320), 0, Math.PI * 2)
      ctx.strokeStyle = 'rgba(255,255,255,0.95)'
      ctx.lineWidth = 1.6
      ctx.stroke()
    }
  }
}
