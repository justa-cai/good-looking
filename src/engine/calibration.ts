/**
 * 校准数据：各指标的人群分布，以及由此确定的理想值与容差。
 *
 * ============================ 样本说明 ============================
 * 样本：37 张正脸证件照（美国国会官方肖像，公有领域），共 30 人，
 *       其中 9 人有 2-3 张不同照片 —— 用来分离「人与人差异」和「照片噪声」。
 * 采集：2026-09-12，MediaPipe FaceLandmarker（float16, 478 点），
 *       输入缩放到最长边 1280px，CPU 后端。
 * 复现：跑 `tmp/calibrate.js`（临时脚本，见 CLAUDE.md「几何指标的校准」）。
 *
 * ============================ 已知局限 ============================
 * 1. 样本全是成年人、以中年为主、几乎都是正脸证件照，以欧美族裔为主。
 *    换成别的年龄段、族裔、拍摄条件，分布很可能不同。
 * 2. 这批人是随机的公众人物，**不代表「好看」的分布**。由它得出的理想值
 *    是「人群平均」，不是「更美」。所以这些分数衡量的是「像不像普通人」，
 *    不是吸引力。界面上必须讲清楚，不要含糊。
 * 3. 样本量 37 只够定出粗略的中心与容差（约 ±20% 的精度），
 *    不足以刻画分布尾部。别拿它做很细的百分位排名。
 *
 * ============================ 为什么有 empirical ============================
 * 4 个指标存在经典约定，但实测均值偏离约定超过 2 个标准差。例如
 * 「两内眼角间距 = 一只眼的宽度」（即 interCanthal / IPD = 0.5）实测 0.541。
 * 30 个人同方向、同幅度偏离，不可能来自人本身，只能来自**测量系统**：
 * MediaPipe 的眼角关键点落在睑裂开口的角上，比解剖学的眼角靠内，
 * 于是眼宽被量小、内眼角间距被量大。鼻翼点(48/278)同理落在鼻翼外缘之外。
 *
 * 所以这些指标改用人群均值当理想值，标为 `empirical`。代价是它衡量的是
 * 「接近人群平均」而非「符合公认比例」—— 界面上要如实区分。
 * 若将来换成解剖学更准的关键点，这些项必须重新评估能否改回 canon。
 *
 * ============================ 怎么改这张表 ============================
 * 数值不是手填的，是用 metrics.ts 的测量式在样本上跑出来后统计得到的。
 * 改动任何测量式（哪怕只是换一个关键点）都要重跑校准，
 * 否则理想值与实际测量对不上，分数会整体偏移而没人察觉。
 */

import type { IdealSource, Reliability } from './types.ts'

export interface Calibration {
  /** 理想值。canon 时等于经典约定值，其余等于人群均值。 */
  readonly ideal: number
  /** 偏离多少时得 50 分（= 人群总标准差 × 1.5） */
  readonly tolerance: number
  readonly idealSource: IdealSource
  readonly reliability: Reliability
  /** 人群均值 */
  readonly mean: number
  /** 人群总标准差 */
  readonly sd: number
  /** 组内（同一人不同照片）标准差，代表测量 + 拍摄噪声 */
  readonly withinSd: number
  /** 噪声占总方差的比例，0-1。越大表示这个指标越主要在被噪声支配。 */
  readonly noiseShare: number
}

/** 由噪声占比判定可信度：<20% 高，<30% 中，其余低。 */
function reliabilityOf(noiseShare: number): Reliability {
  if (noiseShare < 0.2) return 'high'
  if (noiseShare < 0.3) return 'medium'
  return 'low'
}

function cal(
  mean: number,
  sd: number,
  withinSd: number,
  ideal: number,
  idealSource: IdealSource,
): Calibration {
  const noiseShare = sd > 0 ? (withinSd * withinSd) / (sd * sd) : 0
  return {
    mean,
    sd,
    withinSd,
    noiseShare,
    ideal,
    /**
     * 容差取 1.5 个标准差：偏离 1.5σ 得 50 分，3σ 接近 0 分。
     * 用 1.5 而不是 1.0，是为了避免分数分布全挤在 60-100 区间 ——
     * 若容差 = 1σ，一半的人每项都会低于 50 分，看起来像在惩罚普通人。
     * （这里假设各类指标大致正态；样本量小，没验证过尾部。）
     */
    tolerance: sd * 1.5,
    idealSource,
    reliability: reliabilityOf(noiseShare),
  }
}

/**
 * key 与 metrics.ts 的 MEASURES / METRIC_META 一一对应，三处要同步改。
 */
export const CALIBRATION: Record<string, Calibration> = {
  // 经典三等分要求上庭 = 中庭。实测 1.026，只差 0.51σ —— 经典约定成立，
  // 也说明 10(发际) 与 168(鼻根) 在这条线上是可信的。
  upperOverMiddle: cal(1.0262, 0.0518, 0.0228, 1.0, 'canon'),

  // 实测 1.514，经典三等分要求 1.0，差 3.23σ —— 不可能改回 1.0。
  // 152 是下巴的**软组织**下缘，比解剖学的颏下点低，下庭天然更大。
  // 噪声占比 40%，是全部指标里最不可信的一项，界面要标出来。
  lowerOverMiddle: cal(1.514, 0.1591, 0.1008, 1.514, 'empirical'),

  // 无公认标准值，用人群均值。噪声占比 12.6%，是本项目最可信的测量。
  faceWidthOverHeight: cal(0.8358, 0.0419, 0.0149, 0.8358, 'typical'),

  // 同样没有公认标准。标准差很小（0.018），所以容差也小，
  // 对这个指标要敏感一些；噪声占比 21.6%，中等可信。
  jawOverFaceWidth: cal(0.8198, 0.0184, 0.0086, 0.8198, 'typical'),

  // 经典约定 0.5（「两眼相距一只眼的宽度」），实测 0.541，差 2.00σ。
  // 原因见文件头对眼角关键点偏移的解释。
  interCanthalOverIPD: cal(0.5406, 0.0203, 0.0085, 0.5406, 'empirical'),

  // 经典约定 1.0，实测 1.178，差 2.15σ —— 鼻翼关键点落在鼻翼外缘之外。
  noseOverInterCanthal: cal(1.1779, 0.0826, 0.0338, 1.1779, 'empirical'),

  // 经典约定 1.5，实测 1.577，只差 0.69σ —— 经典约定成立。
  mouthOverNose: cal(1.5770, 0.1116, 0.0600, 1.5, 'canon'),
}
