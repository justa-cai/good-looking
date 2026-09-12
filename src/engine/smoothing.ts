/**
 * 实时读数的滚动窗口平滑。
 *
 * 为什么必须平滑：文档实测同一人不同照片的综合分极差 0.7–15.4 分，姿态极差与
 * 分数极差的相关系数 0.91，单张照片本身就有 ±5–8 分噪声。视频帧还多出压缩伪影、
 * 运动模糊、自动曝光漂移。逐帧原始分直接上屏会以肉眼可见的频率乱跳，而文档明确
 * 要求「不要把个位数的分差当成有意义的差异来解读或展示」。所以这里把最近若干帧
 * 压成一个值 + 一个不确定度，**原始分永远不上屏**。
 *
 * 三个选择及理由：
 *
 * - **中位数**而不是均值：姿态毛刺会产生离群帧，均值会被拽走。
 * - **MAD × 1.4826** 而不是 p90−p10 当离散度：n=30 时 p90 是第 3 大样本，被尾巴
 *   主导；而 MAD 是稳健标准差，**与文档里 ±5~8 分的噪声同量纲**，所以界面可以
 *   直接写「±X 分」，不必另造一把「稳定度」尺子。
 * - **中位数再过一道短 EMA**：裸滚动中位数是阶梯状输出，值只在半窗翻转时才跳，
 *   看起来像卡顿。EMA 的代价是几百毫秒的额外滞后，相对 2.5s 的窗口可以接受。
 *
 * 纯函数式的工厂，无 DOM、无依赖，与 engine/ 其余部分一致。
 */

/**
 * 一个待平滑的样本。
 *
 * 姿态和瞳距不是用来算分的，是用来**决定要不要清空窗口**的：
 * 缓慢转头会让中位数从 80 走到 70（文档实测 0°/10°/20°/-10° 分别得
 * 80.1/81.2/76.7/70.0），而每帧单独看都合法、离散度也小 —— 不设防的话
 * 界面会一直显示「稳定」，用户会以为自己的脸变了。
 */
export interface LiveSample {
  /** 这一帧的综合分 */
  readonly score: number
  /**
   * yaw/pitch/roll 里绝对值最大的那个，单位度。
   * 由调用方用 face/pose.ts 的 readHeadPose 算好传进来，这里不碰关键点。
   */
  readonly maxPoseAngle: number
  /** 这一帧的瞳距（像素），用于检测人是否前后移动了 */
  readonly ipd: number
  /** 采样时间，毫秒。必须与 read() 的入参同源 */
  readonly t: number
}

export interface LiveReading {
  /** 平滑后的综合分。**显示时必须取整** —— 文档禁止展示小数分 */
  readonly score: number
  /** 稳健标准差（MAD×1.4826），可直接当作「±X 分」报出去 */
  readonly uncertainty: number
  /** 参与平滑的样本数 */
  readonly n: number
  /** 是否已经稳定：样本够 + 离散度小 */
  readonly stable: boolean
  /** 最近有没有新样本进来。旧值该变暗而不是继续自信地亮着 */
  readonly stale: boolean
}

/** 窗口内最多保留的样本数 */
const MAX_SAMPLES = 30
/** 窗口内样本的最长寿命（毫秒）。只按样本数封顶的话，慢设备上窗口会跨越十几秒 */
const MAX_AGE_MS = 2500
/** 少于这个样本数不出数，只显示「正在测量…」 */
const MIN_SAMPLES = 8
/** 样本数要达到窗口容量的这个比例，才可能判为稳定 */
const STABLE_FRACTION = 0.6
/**
 * 判为稳定的稳健标准差上限（分）。
 *
 * ⚠️ 这个值是**待标定**的：和 sharpness.ts 的 4.0、calibration.ts 的容差一样，
 * 应当由夹具实测的分布定出来，不能凭直觉填。实测方法见 CLAUDE.md
 * 「摄像头实时推理」一节：静止夹具给出抖动地板，缓慢旋转夹具给出真实运动下的分布。
 */
const STABLE_SIGMA = 3.0
/** 输出 EMA 的系数。0.3 约合 3-4 个样本收敛，约 0.3s */
const EMA_ALPHA = 0.3
/**
 * 允许进窗口的姿态上限（度）。
 *
 * ⚠️ 刻意远紧于 quality.ts 的 20° 拒判线 —— 文档自己说 20° 是「安全网，不是
 * 精确刻度」。20° 内的转头照样能把分数挪 10 分（见上面 80.1/76.7/70.0 那组数），
 * 所以那个网用来拒判坏照片可以，用来保证实时读数可比就不够。
 */
const TIGHT_POSE_DEG = 6
/** 瞳距相对变化超过这个比例就认为人前后移动了，窗口作废 */
const IPD_DRIFT = 0.05
/** 超过这个时间没有新样本就标记为陈旧 */
const STALE_MS = 800

export interface LiveSmoother {
  /**
   * 送入一帧。姿态超出紧带、或瞳距明显变化时会**清空窗口**而不是仅仅不入栈 ——
   * 留着旧样本正是让中位数被缓慢转头带着走的那个原因。
   *
   * @returns 这一帧是否被采纳
   */
  push(sample: LiveSample): boolean
  /** 读取当前读数。样本不足（< MIN_SAMPLES）时返回 null，调用方应显示「正在测量…」 */
  read(now: number): LiveReading | null
  /** 清空。换照片、重新开摄像头、页面切到后台时调用 */
  reset(): void
  /** 窗口内当前样本数，供调试与界面提示用 */
  size(): number
}

export function createLiveSmoother(): LiveSmoother {
  /** 与原始样本同时入栈的时间戳，用于按寿命淘汰 */
  let samples: LiveSample[] = []
  let smoothed: number | null = null

  function reset(): void {
    samples = []
    smoothed = null
  }

  function push(sample: LiveSample): boolean {
    if (!Number.isFinite(sample.score)) return false

    // 姿态出带：清空而不是跳过。见 TIGHT_POSE_DEG 的说明
    if (!(sample.maxPoseAngle <= TIGHT_POSE_DEG)) {
      reset()
      return false
    }

    // 人前后移动了：尺度基准变了，旧样本和新样本不可比
    const ref = samples[0]
    if (ref && Math.abs(sample.ipd - ref.ipd) / ref.ipd > IPD_DRIFT) {
      reset()
    }

    samples.push(sample)

    // 按寿命和数量双重淘汰
    const cutoff = sample.t - MAX_AGE_MS
    while (samples.length > 0 && samples[0]!.t < cutoff) samples.shift()
    while (samples.length > MAX_SAMPLES) samples.shift()

    return true
  }

  function read(now: number): LiveReading | null {
    if (samples.length < MIN_SAMPLES) return null

    const values = samples.map((s) => s.score).sort((a, b) => a - b)
    const center = medianSorted(values)
    const sigma = madSigma(values, center)

    // 中位数是阶梯状的，再过一道 EMA 让它连续
    smoothed = smoothed === null ? center : smoothed + EMA_ALPHA * (center - smoothed)

    const last = samples[samples.length - 1]!
    return {
      score: smoothed,
      uncertainty: sigma,
      n: samples.length,
      stable: samples.length >= Math.ceil(MAX_SAMPLES * STABLE_FRACTION) && sigma <= STABLE_SIGMA,
      stale: now - last.t > STALE_MS,
    }
  }

  return { push, read, reset, size: () => samples.length }
}

/** 已排序数组的中位数 */
function medianSorted(sorted: readonly number[]): number {
  const mid = sorted.length >> 1
  if (sorted.length === 0) return Number.NaN
  return sorted.length % 2 === 1 ? sorted[mid]! : (sorted[mid - 1]! + sorted[mid]!) / 2
}

/**
 * 稳健标准差：MAD × 1.4826。
 *
 * 1.4826 是让它在正态分布下与标准差一致的那个系数。这里不直接报标准差，
 * 是因为标准差被离群帧主导，而实时场景里离群帧总会来几个。
 */
function madSigma(sorted: readonly number[], center: number): number {
  const deviations = sorted.map((v) => Math.abs(v - center)).sort((a, b) => a - b)
  return medianSorted(deviations) * 1.4826
}
