/**
 * 实时取景的逐帧判定。
 *
 * 这一层回答「这一帧意味着什么」：分数是多少、要不要信、该提示用户做什么。
 * 它不碰 DOM —— 画什么、怎么排版是 `ui/live.ts` 的事。这样它和 engine/ 一样
 * 是纯逻辑，可以脱离界面单独验证（夹具探针就是直接调它）。
 *
 * 三件必须一起做的事，缺一个结论就不成立：
 *
 * 1. **先过 quality.ts，再进平滑。** 实时路径调 `checkQuality(faces)` 时
 *    **不传位图**，于是它天然跳过模糊检测（`quality.ts` 里那个 `if (image)`），
 *    一行都不用改。姿态、人脸大小、出画这几关照常生效。
 * 2. **模糊单独抽帧测，只当提示、不拒判。** 见下面的 SHARPNESS_INTERVAL_MS。
 * 3. **原始分永不上屏。** 上屏的只有 `LiveReading`（中位数 + 稳健 σ + 稳定徽章）。
 */

import { analyze } from '../engine/index.ts'
import {
  createLiveSmoother,
  POSE_TRUST_DEG,
  type LiveReading,
  type LiveSmoother,
} from '../engine/smoothing.ts'
import { detectFacesFromVideo } from '../face/landmarker.ts'
import { readHeadPose, type HeadPose } from '../face/pose.ts'
import { checkQuality, QUALITY_THRESHOLDS, type QualityIssue } from '../face/quality.ts'
import { measureSharpness } from '../face/sharpness.ts'
import type { FaceLandmarks, Point3 } from '../face/types.ts'

/**
 * 抽帧测模糊的间隔（毫秒）。
 *
 * 模糊是本项目里唯一「不会自己暴露」的坏输入（见 CLAUDE.md）：它不动关键点的
 * 相对位置，所以分数照样给得出来，只是每个点抖得更厉害。而**稳定度指示对它
 * 结构性失明** —— 一个昏暗房间里静止不动的人会得到一个又稳又错的分。
 * 所以实时模式不能把这一关砍掉，只能摊薄成本：它要读原始像素、要建一张位图，
 * 但每 1 秒一次摊下来远低于 0.5ms/帧。
 *
 * 用时间而不是帧数来节流：检测速率在不同设备上能差 3 倍，按帧数算的话
 * 慢设备上会变成 0.3Hz、快设备上 2Hz。
 */
const SHARPNESS_INTERVAL_MS = 1000

/**
 * 清晰度档位翻转需要的连续一致次数。配合 1Hz 的采样约合 2 秒停留 ——
 * 否则在阈值附近会以 1Hz 的频率闪提示。
 */
const SHARPNESS_DWELL_SAMPLES = 2

/**
 * 清晰度档位翻转的死区（分数）。
 *
 * 阈值是在 37 张**专业肖像**上标定的，而网络摄像头整体更软、还会自动增益，
 * 实测值在阈值上下晃是常态。没有死区的话提示会一直闪。
 * 代价是档位会「粘」在旧的判断上一会儿 —— 这是刻意的权衡。
 */
const SHARPNESS_HYSTERESIS = 0.5

/**
 * 叠加层用的关键点 EMA 系数。
 *
 * 每帧画 478 个原始点会明显抖动，而用户对画面抖动的怀疑远大于对数字的怀疑。
 * 0.4 约合 2 帧的滞后（~170ms），来得及看清又明显更稳。
 *
 * ⚠️ 平滑只作用于**画出来的点**，分数用的始终是这一帧的原始关键点。
 * 叠加层是取景辅助，不是测量结果。
 */
const DISPLAY_ALPHA = 0.4

/** 清晰度档位。`unknown` 表示还没测过（不等于清晰） */
export type SharpnessState = 'unknown' | 'ok' | 'soft' | 'bad'

export type GuidanceKind =
  | 'no-face'
  /** 姿态超出 quality 的拒判线 */
  | 'pose'
  | 'too-far'
  | 'edge'
  | 'blurry'
  /** 有效样本还不够，正在攒 */
  | 'measuring'
  /** 拿不到姿态数据之类的、多半是我们这边的问题 */
  | 'settling'
  | 'error'

export interface Guidance {
  readonly kind: GuidanceKind
  readonly text: string
  /** info 只是说明，warn 要用户动手改 */
  readonly tone: 'info' | 'warn'
}

export interface LiveFrame {
  /**
   * 这一帧检出的脸。**不管有没有过质量校验** —— 被拒判时它仍然有值。
   *
   * 区分「没检到脸」和「检到了但不合格」是必须的：两者给用户的处置完全不同
   * （前者让他把脸放进画面，后者要告诉他到底哪一项不合格），而 `checkQuality`
   * 返回的 `face` 在拒判时是 null，拿它当判据就会把「脸太小」说成「请把脸放进画面中间」，
   * 同时叠加层一个字都画不出来 —— 用户看到的是黑屏加一句答非所问的提示。
   */
  readonly face: FaceLandmarks | null
  /** 真正拿去打分的那张脸。被拒判或打分失败时为 null */
  readonly scoredFace: FaceLandmarks | null
  /** 叠加层用的点：EMA 平滑过的坐标。没检到脸时为 null */
  readonly displayPoints: readonly Point3[] | null
  /** 平滑后的读数。样本不足时为 null —— 此时界面显示「正在测量」 */
  readonly reading: LiveReading | null
  /** 这一帧的**原始**分。只用于调试读数，不要上屏 */
  readonly rawScore: number | null
  readonly pose: HeadPose | null
  /** 瞳距（像素） */
  readonly ipd: number
  /** 被拒判时的第一条原因 */
  readonly reject: QualityIssue | null
  /** 取景提示，已带停留 */
  readonly guidance: Guidance | null
  readonly sharpnessState: SharpnessState
  /** 最近一次测到的归一化清晰度，没测过为 null */
  readonly sharpness: number | null
  /** 实测帧率（EMA） */
  readonly fps: number
}

export interface LiveSession {
  /** 处理一帧。检测本身抛错时会把错误抛给调用方，由帧循环计数 */
  frame(now: number, deltaMs: number): LiveFrame
  /** 清空所有累积状态：换流、切回前台、进入冻结时调用 */
  reset(): void
}

export function createLiveSession(video: HTMLVideoElement): LiveSession {
  const smoother: LiveSmoother = createLiveSmoother()

  /** 叠加层专用的 EMA 缓冲，跟着点数走 */
  let emaX: Float64Array | null = null
  let emaY: Float64Array | null = null

  let lastSharpnessAt = Number.NEGATIVE_INFINITY
  /** 位图快照是异步的，同一时间只允许一个在飞 */
  let sharpnessBusy = false
  let sharpnessState: SharpnessState = 'unknown'
  let sharpnessValue: number | null = null
  let sharpnessCandidate: SharpnessState | null = null
  let sharpnessHits = 0
  /** 只报一次，避免每帧刷屏 */
  let analyzeErrorReported = false

  let fps = 0

  // 提示的停留状态。见 commitGuidance
  let shownGuidance: Guidance | null = null
  let pendingGuidance: Guidance | null = null
  let pendingGuidanceAt = 0

  function reset(): void {
    smoother.reset()
    emaX = null
    emaY = null
    lastSharpnessAt = Number.NEGATIVE_INFINITY
    sharpnessBusy = false
    sharpnessState = 'unknown'
    sharpnessValue = null
    sharpnessCandidate = null
    sharpnessHits = 0
    shownGuidance = null
    pendingGuidance = null
    fps = 0
  }

  /**
   * 提示的停留：同一句话要连着说够 GUIDANCE_DWELL_MS 才上屏。
   *
   * 取景提示是从每帧的测量结果推出来的，而测量结果本身在阈值附近会抖
   * （「脸太小」和「刚好够大」之间只有几个像素的差别）。不设停留的话
   * 文案会以 12Hz 在两句话之间闪，比不提示还糟。
   */
  const GUIDANCE_DWELL_MS = 400

  function commitGuidance(next: Guidance | null, now: number): Guidance | null {
    const sameAsShown = next?.text === shownGuidance?.text
    if (sameAsShown) {
      pendingGuidance = null
      return shownGuidance
    }

    if (pendingGuidance?.text === next?.text) {
      if (now - pendingGuidanceAt >= GUIDANCE_DWELL_MS) {
        shownGuidance = next
        pendingGuidance = null
      }
    } else {
      pendingGuidance = next
      pendingGuidanceAt = now
    }
    return shownGuidance
  }

  /** 把点位刷新到这一帧。点数变了就重来，不给新脸叠上旧脸的残影 */
  function smoothPoints(face: FaceLandmarks): readonly Point3[] {
    const n = face.pixels.length
    if (!emaX || !emaY || emaX.length !== n) {
      emaX = new Float64Array(n)
      emaY = new Float64Array(n)
      for (let i = 0; i < n; i++) {
        const p = face.pixels[i]!
        emaX[i] = p.x
        emaY[i] = p.y
      }
    } else {
      for (let i = 0; i < n; i++) {
        const p = face.pixels[i]!
        emaX[i]! += DISPLAY_ALPHA * (p.x - emaX[i]!)
        emaY[i]! += DISPLAY_ALPHA * (p.y - emaY[i]!)
      }
    }

    const out: Point3[] = new Array(n)
    for (let i = 0; i < n; i++) {
      out[i] = { x: emaX[i]!, y: emaY[i]!, z: face.pixels[i]!.z }
    }
    return out
  }

  /**
   * 抓一帧位图测清晰度。
   *
   * 关掉位图是必须的：不 close 的话每张位图都会占住一份解码后的像素，
   * 一秒一张，几分钟就是几百 MB。
   *
   * ⚠️ 用的是**上一次检测那一帧**的关键点，而位图是现在这一帧。30fps 下相差
   * 一帧（33ms），对「外扩 12% 的包围盒重采样到 160×160」这个尺度没有影响 ——
   * 但这也意味着它是**粗测**，不适合拿去判断有没有轻微发虚。
   */
  async function sampleSharpness(face: FaceLandmarks): Promise<void> {
    let bitmap: ImageBitmap | null = null
    try {
      // 不带 resize 选项：位图尺寸必须等于 video.videoWidth/videoHeight，
      // 因为关键点就在那个坐标系里。缩了的话包围盒会落到别的地方去
      bitmap = await createImageBitmap(video)
      const measured = measureSharpness(bitmap, face)
      if (measured) observeSharpness(measured.sharpness)
    } catch {
      // 快照失败（尺寸还是 0、位图创建被拒）只是这一项没测，不影响出分
    } finally {
      bitmap?.close()
      sharpnessBusy = false
    }
  }

  function observeSharpness(value: number): void {
    sharpnessValue = value
    const target = levelOf(value)

    if (target === sharpnessState) {
      sharpnessCandidate = null
      sharpnessHits = 0
      return
    }
    if (!beyondDeadband(sharpnessState, target, value)) {
      sharpnessCandidate = null
      sharpnessHits = 0
      return
    }

    if (sharpnessCandidate === target) sharpnessHits++
    else {
      sharpnessCandidate = target
      sharpnessHits = 1
    }

    if (sharpnessHits >= SHARPNESS_DWELL_SAMPLES) {
      sharpnessState = target
      sharpnessCandidate = null
      sharpnessHits = 0
    }
  }

  function frame(now: number, deltaMs: number): LiveFrame {
    if (deltaMs > 0) {
      const instant = 1000 / deltaMs
      fps = fps === 0 ? instant : fps + 0.15 * (instant - fps)
    }

    const faces = detectFacesFromVideo(video, now)
    const quality = checkQuality(faces)

    // VIDEO 实例的 numFaces 是 1，所以最多一张。但即使将来改成多张，
    // 这里也只看第一张 —— 实时模式只跟最主要的那张脸
    const face = faces[0] ?? null
    const pose = face ? readHeadPose(face.transform) : null

    let scoredFace: FaceLandmarks | null = null
    let rawScore: number | null = null
    let analyzeError: unknown = null

    if (quality.acceptable && quality.face && pose) {
      scoredFace = quality.face
      try {
        rawScore = analyze(quality.face.pixels).score
      } catch (err) {
        // 理论上过不了 quality 才会走到这里（瞳距无效之类），但真走到时
        // 不能让整个循环崩掉 —— 记一次日志，这一帧当作没测出来
        analyzeError = err
        if (!analyzeErrorReported) {
          analyzeErrorReported = true
          console.error('[good-looking] 实时打分失败：', err)
        }
      }

      if (rawScore !== null) {
        smoother.push({
          score: rawScore,
          maxPoseAngle: Math.max(Math.abs(pose.yaw), Math.abs(pose.pitch), Math.abs(pose.roll)),
          ipd: quality.ipd,
          t: now,
        })
      }
    }

    // 没有脸 / 被拒判时**不清空窗口**，只是不再往里推。让样本按 2.5s 的寿命
    // 自然老死：短暂遮挡（手挥过、眨一下眼引起的整帧漏检）不该让数字闪掉，
    // 而 read() 会在 800ms 后把读数标成 stale、2.5s 后变成 null。
    const reading = smoother.read(now)

    // 清晰度和叠加层都用**检出的**那张脸，与有没有过质量校验无关：
    // 「脸太小」时用户更需要看到关键点到底落在哪儿
    if (face) maybeSampleSharpness(face, now)

    const displayPoints = face ? smoothPoints(face) : null

    const reject = quality.issues.find((i) => i.severity === 'reject') ?? null
    const guidance = commitGuidance(
      deriveGuidance({
        face,
        reject,
        reading,
        sharpnessState,
        ipd: quality.ipd,
        analyzeError,
      }),
      now,
    )

    return {
      face,
      scoredFace,
      displayPoints,
      reading,
      rawScore,
      pose,
      ipd: quality.ipd,
      reject,
      guidance,
      sharpnessState,
      sharpness: sharpnessValue,
      fps,
    }
  }

  function maybeSampleSharpness(face: FaceLandmarks, now: number): void {
    if (sharpnessBusy) return
    if (now - lastSharpnessAt < SHARPNESS_INTERVAL_MS) return
    lastSharpnessAt = now
    sharpnessBusy = true
    void sampleSharpness(face)
  }

  return { frame, reset }
}

/** 无死区的原始分档 */
function levelOf(value: number): SharpnessState {
  const { minSharpness, warnSharpness } = QUALITY_THRESHOLDS
  if (value < minSharpness) return 'bad'
  if (value < warnSharpness) return 'soft'
  return 'ok'
}

/**
 * 死区：往**坏**的方向翻转要真的低于阈值，往**好**的方向翻转要高过阈值。
 *
 * 只有相邻档位之间才谈得上死区，`unknown` 不受限制（第一次测量就该定档）。
 */
function beyondDeadband(from: SharpnessState, to: SharpnessState, value: number): boolean {
  const { minSharpness, warnSharpness } = QUALITY_THRESHOLDS
  const H = SHARPNESS_HYSTERESIS

  if (from === 'unknown') return true
  if (from === 'ok' && to === 'soft') return value < warnSharpness - H
  if (from === 'soft' && to === 'ok') return value > warnSharpness + H
  if (from === 'soft' && to === 'bad') return value < minSharpness - H
  if (from === 'bad' && to === 'soft') return value > minSharpness + H
  // 跳档（ok ↔ bad）：由停留次数兜住，这里放行
  return true
}

interface GuidanceInput {
  /** **检出的**脸，不是通过校验的那张。见 LiveFrame.face */
  readonly face: FaceLandmarks | null
  readonly reject: QualityIssue | null
  readonly reading: LiveReading | null
  readonly sharpnessState: SharpnessState
  readonly ipd: number
  readonly analyzeError: unknown
}

/**
 * 取景提示。**一次只说一句**，按「越是拦着出分的越靠前」排序。
 *
 * 顺序不是随手定的：无脸 → 拒判 → 清晰度 → 样本不足 → 各种「能出分但不太好」。
 * 用户同一时间只可能照着一句话去调，把三条一起显示出来等于让他自己排优先级。
 */
function deriveGuidance(input: GuidanceInput): Guidance | null {
  const { face, reject, reading, sharpnessState, ipd, analyzeError } = input

  if (!face) {
    return {
      kind: 'no-face',
      text: '请把脸放进画面中间，正对镜头',
      tone: 'warn',
    }
  }

  if (analyzeError !== null) {
    return { kind: 'error', text: '这一帧没能算出分数，请保持不动', tone: 'warn' }
  }

  if (reject) {
    switch (reject.code) {
      case 'pose-extreme':
        return { kind: 'pose', text: '头偏得有点多，请正对镜头', tone: 'warn' }
      case 'face-too-small':
        return { kind: 'too-far', text: '请离镜头近一点，让脸占满中间的取景框', tone: 'warn' }
      case 'face-borders-image':
        return { kind: 'edge', text: '请退后一点，让整个头都在画面里', tone: 'warn' }
      case 'multiple-faces':
        return { kind: 'error', text: '画面里不止一张脸，请只留一个人', tone: 'warn' }
      case 'no-pose-data':
        // 这多半是我们这边没拿到变换矩阵，不是用户的问题，措辞别推给用户
        return { kind: 'settling', text: '正在重新定位，请稍等一下', tone: 'info' }
      case 'landmarks-incomplete':
        return { kind: 'settling', text: '关键点还不完整，请把脸摆正一点', tone: 'warn' }
      default:
        return { kind: 'error', text: reject.message, tone: 'warn' }
    }
  }

  // 到这儿说明这一帧是合格的，只是可能还不够好
  if (sharpnessState === 'bad') {
    return {
      kind: 'blurry',
      text: '画面发虚，请检查对焦和光线 —— 糊的照片照样给分，但分数不可信',
      tone: 'warn',
    }
  }

  if (!reading) {
    // 分两种情况：样本还不够；或者窗口刚被姿态漂移清空又得重新攒。
    // 对用户来说处置是同一件事（端住别动），所以同一句话
    return { kind: 'measuring', text: '正在测量，请端住不动…', tone: 'info' }
  }

  return warnFor(reading, sharpnessState, ipd)
}

/** 已经能出分时的次要提示。同样只给一条 */
function warnFor(reading: LiveReading, sharpnessState: SharpnessState, ipd: number): Guidance | null {
  if (reading.maxPoseAngle > POSE_TRUST_DEG) {
    return {
      kind: 'pose',
      text: `头还偏着约 ${reading.maxPoseAngle.toFixed(0)}°，这时分数只能粗略看`,
      tone: 'info',
    }
  }
  if (sharpnessState === 'soft') {
    return { kind: 'blurry', text: '画面略微发虚，读数会跳得比清晰时大', tone: 'info' }
  }
  // 瞳距在 50~100px 之间：能出分，但关键点精度已经开始影响结果。
  // 摄像机语境下这几乎总是「坐得离镜头远」，所以直说距离而不是说「照片里脸小」
  if (ipd > 0 && ipd < QUALITY_THRESHOLDS.warnIpd) {
    return { kind: 'too-far', text: '离镜头再近一点会更准', tone: 'info' }
  }
  if (reading.stale) {
    return { kind: 'measuring', text: '暂时没跟上面部，读数正在变旧', tone: 'info' }
  }
  return null
}
