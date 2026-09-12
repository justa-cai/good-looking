/**
 * 指标定义：怎么从关键点算出测量值。
 *
 * 这里的测量方式必须与 calibration.ts 里的分布来自同一套代码 ——
 * 校准数据是用本文件的函数在样本上跑出来的，改动任何一个测量式都要重跑校准。
 * 详见 CLAUDE.md「几何指标的校准」。
 *
 * 两条通用约定：
 *
 * 1. 所有长度都以**瞳距**为单位。瞳距由虹膜中心给出，是全脸最稳定的一段，
 *    而且不受发型影响。用面高当尺度会被发际线检测带偏，用面宽会被头发带偏。
 *
 * 2. 纵向比例一律沿「发际→下巴」这条中轴做投影，不直接用 y 之差。
 *    头在画面里歪了几度时，y 之差会被旋转放大，而投影不会。
 */

import { LM } from '../face/indices.ts'
import { axialProgress, dist } from './geometry.ts'
import type { Vec3 } from './types.ts'

/** 测量上下文，避免每个函数都重复算同一批中间量。 */
export interface MeasureContext {
  /** 取第 i 个关键点。越界会抛错而不是返回 undefined —— 调用前已校验数量。 */
  readonly pt: (index: number) => Vec3
  /** 瞳距，作为所有长度的单位 */
  readonly ipd: number
  /** 沿中轴归一化到 [0,1] 的纵向进度，0 = 发际，1 = 下巴 */
  readonly axial: (index: number) => number
}

export function createContext(points: readonly Vec3[]): MeasureContext {
  const pt = (index: number): Vec3 => {
    const p = points[index]
    if (!p) throw new RangeError(`关键点索引 ${index} 越界（共 ${points.length} 个）`)
    return p
  }
  const bottom = pt(LM.chin)
  const top = pt(LM.foreheadTop)
  return {
    pt,
    ipd: dist(pt(LM.irisA), pt(LM.irisB)),
    axial: (index) => axialProgress(pt(index), top, bottom),
  }
}

/** 一个指标的测量式。返回 NaN 表示该图算不出这一项。 */
export type Measure = (c: MeasureContext) => number

const ratio = (a: number, b: number): number => (b === 0 ? Number.NaN : a / b)

/** 指标 id → 测量式。key 必须与 calibration.ts 一致。 */
export const MEASURES = {
  /** 上庭 / 中庭。经典三等分要求两者相等。 */
  upperOverMiddle: (c: MeasureContext) =>
    ratio(
      c.axial(LM.nasion) - c.axial(LM.foreheadTop),
      c.axial(LM.subnasale) - c.axial(LM.nasion),
    ),

  /** 下庭 / 中庭。实测远大于 1，原因见 calibration.ts。 */
  lowerOverMiddle: (c: MeasureContext) =>
    ratio(
      c.axial(LM.chin) - c.axial(LM.subnasale),
      c.axial(LM.subnasale) - c.axial(LM.nasion),
    ),

  /** 面宽 / 面高。 */
  faceWidthOverHeight: (c: MeasureContext) =>
    ratio(dist(c.pt(LM.contourA), c.pt(LM.contourB)), dist(c.pt(LM.foreheadTop), c.pt(LM.chin))),

  /** 下颌宽 / 面宽。越小表示下颌收得越窄。 */
  jawOverFaceWidth: (c: MeasureContext) =>
    ratio(dist(c.pt(LM.jawA), c.pt(LM.jawB)), dist(c.pt(LM.contourA), c.pt(LM.contourB))),

  /** 两内眼角间距 / 瞳距。经典约定是 0.5。 */
  interCanthalOverIPD: (c: MeasureContext) =>
    ratio(dist(c.pt(LM.eyeInnerA), c.pt(LM.eyeInnerB)), c.ipd),

  /** 鼻宽 / 两内眼角间距。经典约定是 1.0。 */
  noseOverInterCanthal: (c: MeasureContext) =>
    ratio(dist(c.pt(LM.alaA), c.pt(LM.alaB)), dist(c.pt(LM.eyeInnerA), c.pt(LM.eyeInnerB))),

  /** 嘴宽 / 鼻宽。经典约定是 1.5。 */
  mouthOverNose: (c: MeasureContext) =>
    ratio(dist(c.pt(LM.mouthCornerA), c.pt(LM.mouthCornerB)), dist(c.pt(LM.alaA), c.pt(LM.alaB))),
} as const satisfies Record<string, Measure>

export type MetricId = keyof typeof MEASURES

/** 界面文案。id 到展示信息的映射，与 MEASURES 一一对应。 */
export const METRIC_META: Record<MetricId, { label: string; description: string }> = {
  upperOverMiddle: {
    label: '上庭 / 中庭',
    description: '发际到鼻根 与 鼻根到鼻底 的高度比。经典三等分要求两者相等，比值偏大表示额头偏长。',
  },
  lowerOverMiddle: {
    label: '下庭 / 中庭',
    description: '鼻底到下巴 与 鼻根到鼻底 的高度比。这一项量的是软组织下巴，正常值本就明显大于 1。',
  },
  faceWidthOverHeight: {
    label: '面宽 / 面高',
    description: '两颧最外缘的宽度 与 发际到下巴的高度之比。表示脸型偏长还是偏宽。',
  },
  jawOverFaceWidth: {
    label: '下颌 / 面宽',
    description: '下颌角一带的宽度占面部最宽处的比例。比值小表示下颌收窄明显。',
  },
  interCanthalOverIPD: {
    label: '眼距 / 瞳距',
    description: '两内眼角间距与瞳距之比。经典约定是 0.5（两眼相距一只眼的宽度），比值偏大表示眼距偏宽。',
  },
  noseOverInterCanthal: {
    label: '鼻宽 / 眼距',
    description: '鼻翼宽度与两内眼角间距之比。经典约定是 1.0，比值偏大表示鼻翼偏宽。',
  },
  mouthOverNose: {
    label: '嘴宽 / 鼻宽',
    description: '嘴角间距与鼻翼宽度之比。经典约定是 1.5，比值偏小表示嘴相对鼻偏窄。',
  },
}

/**
 * 各指标相对上一版测量方式的变动说明保留在此，方便回溯。
 * 2026-09-12：纵向比例由「y 之差」改为「沿中轴投影」，校准数据同步重跑。
 */
