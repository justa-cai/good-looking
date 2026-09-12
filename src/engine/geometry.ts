/**
 * 关键点上的几何工具。纯函数，无副作用。
 */

import type { Vec3 } from './types.ts'

/** 两点欧氏距离，只用 x/y（z 的精度在 MediaPipe 里远不如 x/y，参与运算会引入噪声）。 */
export function dist(a: Vec3, b: Vec3): number {
  return Math.hypot(a.x - b.x, a.y - b.y)
}

/**
 * 三点构成的角（a 为顶点），返回角度制。
 * 用于描述关键点之间的夹角，比直接比坐标稳定。
 */
export function angleAt(a: Vec3, b: Vec3, c: Vec3): number {
  const v1x = b.x - a.x
  const v1y = b.y - a.y
  const v2x = c.x - a.x
  const v2y = c.y - a.y
  const denom = Math.hypot(v1x, v1y) * Math.hypot(v2x, v2y)
  if (denom === 0) return Number.NaN
  const cos = Math.min(1, Math.max(-1, (v1x * v2x + v1y * v2y) / denom))
  return (Math.acos(cos) * 180) / Math.PI
}

/**
 * 沿「鼻根到下巴」这条中轴方向，把点投影到轴上的位置（纵向进度）。
 *
 * 为什么不用「两个点的 y 之差」直接算纵向距离：人一歪头，y 差就会被
 * 旋转放大。投影到真正的中轴方向上，头在画面里转了不影响结果。
 *
 * 返回归一化到 [0,1] 的进度：0 = 落在鼻根高度，1 = 落在下巴高度。
 * 超出这个范围（比如发际线在鼻根之上）会返回负数。
 */
export function axialProgress(point: Vec3, top: Vec3, bottom: Vec3): number {
  const ax = bottom.x - top.x
  const ay = bottom.y - top.y
  const len2 = ax * ax + ay * ay
  if (len2 === 0) return Number.NaN
  return ((point.x - top.x) * ax + (point.y - top.y) * ay) / len2
}

/**
 * 一组左右对称点到中轴的横向偏移是否互相抵消。
 * 返回「单侧平均偏移 / 尺度」，0 表示完全对称。
 *
 * 只看横向（垂直于中轴）分量：歪头造成的整体旋转在横向分量上是同向的，
 * 会互相抵消；而真正的左右不对称是一正一负，不会被抵消。所以这个量
 * 对头部的面内旋转不敏感，适合当姿态/质量的判据。
 */
export function lateralAsymmetry(
  pairs: ReadonlyArray<readonly [Vec3, Vec3]>,
  axisTop: Vec3,
  axisBottom: Vec3,
  scale: number,
): number {
  if (pairs.length === 0 || scale === 0) return Number.NaN
  const ax = axisBottom.x - axisTop.x
  const ay = axisBottom.y - axisTop.y
  const len = Math.hypot(ax, ay)
  if (len === 0) return Number.NaN
  // 中轴的法线方向
  const nx = -ay / len
  const ny = ax / len

  let sum = 0
  for (const [a, b] of pairs) {
    const ta = (a.x - axisTop.x) * nx + (a.y - axisTop.y) * ny
    const tb = (b.x - axisTop.x) * nx + (b.y - axisTop.y) * ny
    sum += Math.abs(ta + tb) / 2
  }
  return sum / pairs.length / scale
}
