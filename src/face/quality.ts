/**
 * 输入质量校验。
 *
 * 目的只有一个：**宁可拒判，也不要静默给一个错的分数。**
 * 几何指标对输入很敏感（见 CLAUDE.md「单张照片的分数有 ±5~8 分的噪声」），
 * 姿态、人脸大小、裁切都会显著改变结果。用户看到分数时无从判断它是怎么来的，
 * 所以不让测的情况必须明确说出来。
 *
 * ⚠️ 这里放弃了一个常见的检查项：**闭眼**。
 * 本来打算用眼睑开合度判断，但实测这个量的个体差异极大
 * （同一批人里从 0.17 到 0.46，相差近 3 倍），闭眼的和睁眼小的分不开，
 * 拿它当拒判条件会大量误伤。而闭眼对几何比例本身几乎没有影响
 * （比例量的是骨架位置，不是眼睛张开的程度），所以不检查反而更正确。
 */

import { LM } from './indices.ts'
import { readHeadPose, type HeadPose } from './pose.ts'
import { measureSharpness } from './sharpness.ts'
import type { FaceLandmarks, Point3 } from './types.ts'

/**
 * 可调的阈值。全部集中在这里，改之前先看下面的依据。
 */
export const QUALITY_THRESHOLDS = {
  /**
   * 头部姿态上限（度）。yaw/pitch/roll 任一超过就拒判。
   *
   * ⚠️ 这是**安全网，不是精确刻度**。20° 的依据：
   * 30 张专业证件照实测的 |yaw| 最大 9.8°、|pitch| 最大 13.0°、|roll| 最大 8.8°，
   * 也就是说 20° 一定能放过正常证件照；而样本里没有真正「拍坏」的照片，
   * 所以**找不到该拒判的那个临界点**，只能取一个明显偏斜才会触发的值。
   *
   * 之所以仍要设这一关：实测同一人不同照片的分数极差与其姿态极差相关系数 0.91，
   * 姿态是分数不稳的主要来源。只是「多大姿态算不能测」缺少数据支撑。
   */
  maxPoseAngle: 20,

  /**
   * 瞳距的像素下限。低于此值关键点定位精度不足以支撑测量。
   *
   * 依据：把一张照片逐步缩小后重测，与原始分数的偏差：
   *   ipd ≥ 100px  |Δ| ≲ 2 分
   *   ipd ≈ 50-70px |Δ| ≈ 4-5 分（开始明显）
   *   ipd ≤ 35px   |Δ| > 6 分，且 27px 时到 13 分
   * 50 是偏差开始失控的位置。
   */
  minIpd: 50,

  /** 低于此值不拒判，但提示用户「人脸偏小，结果可能不准」。 */
  warnIpd: 100,

  /** 人脸包围盒距画面边缘小于此比例时，判为可能被裁切。 */
  croppedMargin: 0.02,

  /**
   * 清晰度下限。低于此值判为明显模糊，拒判。
   *
   * 度量方式见 sharpness.ts（人脸区域归一化后的拉普拉斯方差）。
   * 依据：37 张官方肖像照作为「清晰」样本，再对同一批图施加
   * blur(1/2/3/4px) 合成「糊」样本，各自的分布是：
   *
   *   模糊半径   最小   中位   最大
   *    0px      4.5   10.4   42.3
   *    1px      0.6    5.5   13.3
   *    2px      0.1    2.2    4.8
   *    3px      0.1    1.1    2.5
   *    4px      0.0    0.6    1.4
   *
   * 取 4.0 而不是分布交界处的 4.5，是刻意留的余量：4.5 正好是这批
   * **专业肖像照**里最低的那张，普通用户随手拍的照片整体会比这批更软，
   * 卡在 4.5 会误伤大量其实够用的照片。代价是 blur(2px) 那一档
   * 还剩约 11% 能通过（89% 被拦下），blur(3px) 及以上 100% 拦下。
   *
   * ⚠️ 能力边界：**blur(1px) 这种轻微发虚和原本就清晰的照片分不开**
   * （中位 5.5 vs 10.4，但分布重叠严重）。这一关拦的是「明显糊」，
   * 不是「不够锐」。
   */
  minSharpness: 4.0,

  /** 低于此值不拒判，但提示可能发虚。约 8% 的清晰样本会落进这一档。 */
  warnSharpness: 6.0,
} as const

export type QualityCode =
  | 'no-face'
  | 'multiple-faces'
  | 'face-too-small'
  | 'face-borders-image'
  | 'face-near-edge'
  | 'pose-extreme'
  | 'no-pose-data'
  | 'blurry'
  | 'landmarks-incomplete'

export interface QualityIssue {
  readonly code: QualityCode
  /** reject 会阻止出分；warn 只提示 */
  readonly severity: 'reject' | 'warn'
  /** 可直接展示给用户的中文说明 */
  readonly message: string
  /** 相关原始数值，便于界面展示细节或排查问题 */
  readonly detail?: Readonly<Record<string, number>>
}

export interface QualityReport {
  /** 是否可以直接出分 */
  readonly acceptable: boolean
  readonly issues: readonly QualityIssue[]
  /** 通过了校验的人脸；acceptable 为 false 时是 null */
  readonly face: FaceLandmarks | null
  /** 读到的姿态，没有则为 null */
  readonly pose: HeadPose | null
  /** 瞳距（像素），用于展示与尺寸判断 */
  readonly ipd: number
  /**
   * 归一化清晰度。没传原图（image 参数省略）或读不到像素时为 null，
   * 表示「这一项没测」，不等于「清晰」。
   */
  readonly sharpness: number | null
}

/** 第 i 个点必须存在；越界抛错（调用前已保证数量足够）。 */
function at(points: readonly Point3[], i: number): Point3 {
  const p = points[i]
  if (!p) throw new RangeError(`关键点索引 ${i} 越界`)
  return p
}

function bbox(points: readonly Point3[]) {
  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity
  for (const p of points) {
    if (p.x < minX) minX = p.x
    if (p.x > maxX) maxX = p.x
    if (p.y < minY) minY = p.y
    if (p.y > maxY) maxY = p.y
  }
  return { minX, minY, maxX, maxY }
}

function ipdOf(points: readonly Point3[]): number {
  const a = at(points, LM.irisA)
  const b = at(points, LM.irisB)
  return Math.hypot(a.x - b.x, a.y - b.y)
}

/**
 * 校验一次检测结果。
 *
 * @param faces detectFaces 的返回。空数组表示没检出人脸（正常结果，不是错误）。
 * @param image 这次检测用的位图。**传了才会做模糊检测** —— 模糊必须读原始像素，
 *   关键点里没有这个信息。省略时（例如只做几何校准的脚本）跳过该项检查，
 *   report.sharpness 为 null。
 */
export function checkQuality(
  faces: readonly FaceLandmarks[],
  image?: ImageBitmap,
): QualityReport {
  const issues: QualityIssue[] = []
  const fail = (r: Omit<QualityReport, 'acceptable'>): QualityReport => ({
    acceptable: false,
    ...r,
  })

  if (faces.length === 0) {
    issues.push({
      code: 'no-face',
      severity: 'reject',
      message: '没有检测到人脸。请换一张正面、清晰、光线均匀的照片。',
    })
    return fail({ issues, face: null, pose: null, ipd: 0, sharpness: null })
  }

  if (faces.length > 1) {
    // 不自动挑一张：用户无法知道分数对应的是哪个人，容易被误读。
    issues.push({
      code: 'multiple-faces',
      severity: 'reject',
      message: `检测到 ${faces.length} 张人脸。请上传只包含一个人的照片，或先裁剪。`,
      detail: { count: faces.length },
    })
    return fail({ issues, face: null, pose: null, ipd: 0, sharpness: null })
  }

  const face = faces[0]!
  const points = face.pixels

  // 需要虹膜点（468/473）做尺度基准，478 点模型才有
  if (points.length <= LM.irisB) {
    issues.push({
      code: 'landmarks-incomplete',
      severity: 'reject',
      message: '关键点数据不完整，无法建立尺度基准。',
      detail: { got: points.length, need: LM.irisB + 1 },
    })
    return fail({ issues, face: null, pose: null, ipd: 0, sharpness: null })
  }

  const ipd = ipdOf(points)
  if (!Number.isFinite(ipd) || ipd <= 0) {
    issues.push({
      code: 'landmarks-incomplete',
      severity: 'reject',
      message: '关键点数据异常（瞳距无效）。',
    })
    return fail({ issues, face: null, pose: null, ipd: 0, sharpness: null })
  }

  const pose = readHeadPose(face.transform)
  if (!pose) {
    // 拿不到姿态就无法排除明显的侧脸/仰俯，而这两者会让测量严重失真。
    // 宁可不出分，也不要猜。注意这多半是**我们这边**的问题（模型没给矩阵），
    // 不是用户照片的问题，所以措辞上不要把责任推给用户。
    issues.push({
      code: 'no-pose-data',
      severity: 'reject',
      message: '这次没能读出头部姿态，无法确认拍摄角度是否合适，所以先不给分数。请稍后重试。',
    })
  } else {
    const { maxPoseAngle } = QUALITY_THRESHOLDS
    const worst = Math.max(Math.abs(pose.yaw), Math.abs(pose.pitch), Math.abs(pose.roll))
    if (worst > maxPoseAngle) {
      issues.push({
        code: 'pose-extreme',
        severity: 'reject',
        message:
          `拍摄角度偏得比较多（左右 ${pose.yaw.toFixed(0)}°、` +
          `俯仰 ${pose.pitch.toFixed(0)}°、倾斜 ${pose.roll.toFixed(0)}°）。` +
          `请正面朝着镜头重拍，角度偏差不要超过 ${maxPoseAngle}°。`,
        detail: { yaw: pose.yaw, pitch: pose.pitch, roll: pose.roll, max: maxPoseAngle },
      })
    }
  }

  const { minIpd, warnIpd, croppedMargin } = QUALITY_THRESHOLDS
  if (ipd < minIpd) {
    issues.push({
      code: 'face-too-small',
      severity: 'reject',
      message: `照片里人脸太小（瞳距约 ${Math.round(ipd)} 像素，需要至少 ${minIpd}）。请靠近一点重拍或放大裁剪后上传。`,
      detail: { ipd, minIpd },
    })
  } else if (ipd < warnIpd) {
    issues.push({
      code: 'face-too-small',
      severity: 'warn',
      message: `照片里人脸偏小（瞳距约 ${Math.round(ipd)} 像素），结果可能不够稳定。`,
      detail: { ipd, warnIpd },
    })
  }

  // 模糊必须读原始像素，所以只在调用方给了位图时才测。
  // 测不出来（没有位图 / 取不到 2D 上下文）就记 null 并跳过，不当作「清晰」。
  let sharpness: number | null = null
  if (image) {
    const measured = measureSharpness(image, face)
    if (measured) {
      sharpness = measured.sharpness
      const { minSharpness, warnSharpness } = QUALITY_THRESHOLDS
      const detail = {
        sharpness: measured.sharpness,
        lumaSd: measured.lumaSd,
        minSharpness,
      }
      if (measured.sharpness < minSharpness) {
        issues.push({
          code: 'blurry',
          severity: 'reject',
          message:
            '照片没有对上焦，五官边缘是糊的，关键点位置会抖，算出来的比例不可信。' +
            '请重新对焦拍摄，或换一张更清晰的照片。',
          detail,
        })
      } else if (measured.sharpness < warnSharpness) {
        issues.push({
          code: 'blurry',
          severity: 'warn',
          message: '照片略微发虚，结果的波动可能比清晰照片大一些。',
          detail,
        })
      }
    }
  }

  const box = bbox(points)
  const marginX = face.width * croppedMargin
  const marginY = face.height * croppedMargin
  const outside =
    box.minX < 0 || box.minY < 0 || box.maxX > face.width || box.maxY > face.height
  const nearEdge =
    box.minX < marginX ||
    box.minY < marginY ||
    box.maxX > face.width - marginX ||
    box.maxY > face.height - marginY

  if (outside) {
    issues.push({
      code: 'face-borders-image',
      severity: 'reject',
      message: '人脸超出了画面范围，关键点是被推算出来的，测量不可靠。请上传完整包含头部的照片。',
    })
  } else if (nearEdge) {
    issues.push({
      code: 'face-near-edge',
      severity: 'warn',
      message: '人脸贴近画面边缘，边缘处的关键点可能不准。建议四周多留一些余地。',
    })
  }

  const acceptable = !issues.some((i) => i.severity === 'reject')
  return {
    acceptable,
    issues,
    face: acceptable ? face : null,
    pose,
    ipd,
    sharpness,
  }
}
