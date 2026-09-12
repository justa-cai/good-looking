/**
 * 几何打分引擎的类型。
 *
 * 引擎是纯函数：输入关键点，输出测量值与分数。
 * 不含 DOM、不读全局状态、无第三方依赖 —— 这样它才能被单独验证，
 * 也是这个项目里唯一真正可复用的资产。
 */

/** 与 face/types 的 Point3 结构兼容；引擎不 import 那个模块，避免反向依赖。 */
export interface Vec3 {
  x: number
  y: number
  z: number
}

/**
 * 测量值的理想区间从哪来。
 *
 * - `canon`      经典面部比例约定，且实测人群均值与它接近（差 < 0.75 个标准差）
 * - `empirical`  有经典约定但实测偏离明显，说明是**测量系统**的系统性偏移
 *                （关键点定义与解剖定义不同），于是改用人群均值当理想值
 * - `typical`    本就没有公认标准，理想值 = 人群均值
 *
 * 区分这三者的意义：`canon` 是「符合公认比例」，`empirical`/`typical` 是
 * 「接近人群平均」。两者不是一回事，界面要如实说明。
 */
export type IdealSource = 'canon' | 'empirical' | 'typical'

/** 该指标的测量可信度，由「同一人多张照片」的组内噪声占比决定。 */
export type Reliability = 'high' | 'medium' | 'low'

export interface MetricResult {
  /** 稳定标识，改名前先想清楚是否会被外部使用 */
  readonly id: string
  /** 面向用户的中文名 */
  readonly label: string
  /** 一句话说明在量什么 */
  readonly description: string
  /** 实测值（比值或角度，无量纲；长度类一律以瞳距为单位） */
  readonly value: number
  /** 理想值 */
  readonly ideal: number
  /** 偏离多少时得 50 分 */
  readonly tolerance: number
  /** 样本人群均值。界面用来说「平均是 X，你是 Y」。 */
  readonly populationMean: number
  /** 样本人群标准差 */
  readonly populationSd: number
  readonly idealSource: IdealSource
  readonly reliability: Reliability
  /** 归一化到 0-100 的分项分 */
  readonly score: number
  /** 以标准差为单位的偏离量，带符号；正负的含义见 description */
  readonly deviation: number
}

export interface AreaResult {
  readonly id: string
  readonly label: string
  readonly score: number
  readonly metrics: readonly MetricResult[]
}

export interface AnalysisResult {
  /** 0-100 的综合分 */
  readonly score: number
  readonly areas: readonly AreaResult[]
  /** 所有分项，便于界面或调试直接取用 */
  readonly metrics: readonly MetricResult[]
  /**
   * 本次测量依赖的关键点是否都拿到了。
   * 为 false 时说明关键点数量不足，分数不应展示。
   */
  readonly complete: boolean
  /** 以瞳距为单位的尺度参考，用于界面换算成「多少像素」之类 */
  readonly interpupillaryDistance: number
}

/** 关键点数量不足（例如拿到了 468 点模型而非 478 点）时抛这个。 */
export class InsufficientLandmarksError extends Error {
  constructor(readonly got: number, readonly need: number) {
    super(`关键点不足：需要至少 ${need} 个，实际 ${got} 个`)
    this.name = 'InsufficientLandmarksError'
  }
}
