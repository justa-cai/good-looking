/**
 * 几何打分引擎。
 *
 * 输入一帧的关键点，输出 0-100 的分项分与综合分。
 * 纯函数、无 DOM、无第三方依赖。
 *
 * ⚠️ 这个分数衡量的是**面部比例接近人群平均的程度**，不是吸引力。
 * 见 calibration.ts 文件头对样本与局限的说明，界面上必须如实转述。
 */

import { LM, LANDMARK_COUNT } from '../face/indices.ts'
import { CALIBRATION } from './calibration.ts'
import { createContext, MEASURES, METRIC_META, type MetricId } from './metrics.ts'
import {
  InsufficientLandmarksError,
  type AnalysisResult,
  type AreaResult,
  type MetricResult,
  type Vec3,
} from './types.ts'

export { InsufficientLandmarksError } from './types.ts'
export type { AnalysisResult, AreaResult, MetricResult, IdealSource, Reliability } from './types.ts'
export { METRIC_META } from './metrics.ts'
export type { MetricId } from './metrics.ts'

/**
 * 分项分曲线：偏离 0 得 100，偏离 = tolerance 得 50，再远迅速衰减。
 *
 * 用 exp(-ln2·(Δ/tol)²) 而不是线性扣分，是因为「偏离多少算明显」本身是
 * 连续的：稍微偏一点应该几乎不扣分，明显偏离才该被压下去。
 * 取 ln2 使 tolerance 有明确的含义 —— 刚好及格线的那个偏离量。
 */
export function scoreFromDeviation(deviation: number, tolerance: number): number {
  if (!Number.isFinite(deviation) || !Number.isFinite(tolerance) || tolerance <= 0) return Number.NaN
  return 100 * Math.exp(-Math.LN2 * (deviation / tolerance) ** 2)
}

/** 指标到分组的归属。分组用于界面呈现，综合分对每个分组等权。 */
const AREAS: ReadonlyArray<{ id: string; label: string; metrics: readonly MetricId[] }> = [
  { id: 'proportion', label: '纵向比例', metrics: ['upperOverMiddle', 'lowerOverMiddle'] },
  { id: 'shape', label: '脸型轮廓', metrics: ['faceWidthOverHeight', 'jawOverFaceWidth'] },
  { id: 'eyes', label: '眼距', metrics: ['interCanthalOverIPD'] },
  { id: 'noseMouth', label: '鼻与嘴', metrics: ['noseOverInterCanthal', 'mouthOverNose'] },
]

/**
 * 分析一帧关键点。
 *
 * @param points 像素坐标的关键点数组，至少 478 个（需要虹膜点做瞳距基准）
 * @throws InsufficientLandmarksError 关键点数量不足
 */
export function analyze(points: readonly Vec3[]): AnalysisResult {
  if (points.length < LANDMARK_COUNT) {
    throw new InsufficientLandmarksError(points.length, LANDMARK_COUNT)
  }

  const ctx = createContext(points)
  if (!Number.isFinite(ctx.ipd) || ctx.ipd <= 0) {
    throw new Error('瞳距无效，无法建立尺度基准')
  }

  const byId = new Map<MetricId, MetricResult>()

  for (const id of Object.keys(MEASURES) as MetricId[]) {
    const cal = CALIBRATION[id]
    if (!cal) {
      // 校准表漏了某个指标属于代码缺陷，早失败好过静默少一项
      throw new Error(`指标 ${id} 缺少校准数据`)
    }

    const value = MEASURES[id](ctx)
    const deviation = value - cal.ideal
    const score = scoreFromDeviation(deviation, cal.tolerance)

    byId.set(id, {
      id,
      label: METRIC_META[id].label,
      description: METRIC_META[id].description,
      value,
      ideal: cal.ideal,
      tolerance: cal.tolerance,
      populationMean: cal.mean,
      populationSd: cal.sd,
      idealSource: cal.idealSource,
      reliability: cal.reliability,
      score,
      // 以标准差为单位的带符号偏离，便于界面说「高出/低于平均多少」
      deviation: cal.sd > 0 ? deviation / cal.sd : Number.NaN,
    })
  }

  const areas: AreaResult[] = AREAS.map((area) => {
    const metrics = area.metrics.map((id) => byId.get(id)!).filter(Boolean)
    return {
      id: area.id,
      label: area.label,
      score: mean(metrics.map((m) => m.score)),
      metrics,
    }
  })

  return {
    // 综合分对每个分组等权，而不是对每个指标等权 ——
    // 否则指标多的分组（鼻与嘴）会盖过指标少的分组（眼距）。
    score: mean(areas.map((a) => a.score)),
    areas,
    metrics: [...byId.values()],
    complete: [...byId.values()].every((m) => Number.isFinite(m.score)),
    interpupillaryDistance: ctx.ipd,
  }
}

function mean(xs: readonly number[]): number {
  const valid = xs.filter((x) => Number.isFinite(x))
  if (valid.length === 0) return Number.NaN
  return valid.reduce((a, b) => a + b, 0) / valid.length
}

/** 供界面高亮关键点用：分组到其涉及的关键点索引。 */
export const METRIC_LANDMARKS: Record<MetricId, readonly number[]> = {
  upperOverMiddle: [LM.foreheadTop, LM.nasion, LM.subnasale],
  lowerOverMiddle: [LM.nasion, LM.subnasale, LM.chin],
  faceWidthOverHeight: [LM.contourA, LM.contourB, LM.foreheadTop, LM.chin],
  jawOverFaceWidth: [LM.jawA, LM.jawB, LM.contourA, LM.contourB],
  interCanthalOverIPD: [LM.eyeInnerA, LM.eyeInnerB, LM.irisA, LM.irisB],
  noseOverInterCanthal: [LM.alaA, LM.alaB, LM.eyeInnerA, LM.eyeInnerB],
  mouthOverNose: [LM.mouthCornerA, LM.mouthCornerB, LM.alaA, LM.alaB],
}
