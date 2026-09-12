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
  /** 是否已经稳定：样本够 + 离散度小 + 窗口内姿态没超出可信范围 */
  readonly stable: boolean
  /** 窗口内出现过的最大姿态角（度）。stable 为 false 时可以看它判断是不是角度的问题 */
  readonly maxPoseAngle: number
  /** 最近有没有新样本进来。旧值该变暗而不是继续自信地亮着 */
  readonly stale: boolean
}

/** 窗口内最多保留的样本数 */
const MAX_SAMPLES = 30
/** 窗口内样本的最长寿命（毫秒）。只按样本数封顶的话，慢设备上窗口会跨越十几秒 */
const MAX_AGE_MS = 2500
/** 少于这个样本数不出数，只显示「正在测量…」 */
const MIN_SAMPLES = 8
/**
 * 窗口要填满到这个比例，才可能判为稳定。
 *
 * ⚠️ 判的是**时间跨度**，不是样本数 —— 这一点是实测改过来的，不是设计时想到的。
 * 原来写的是「样本数 ≥ 窗口容量的 60%」，看起来等价，其实把结论绑在了帧率上：
 * 静止夹具实测 12.0 Hz 时能稳定判稳，掉到 **6.2 Hz** 之后同一个画面、同一组分数
 * （±0.0，一个样本都没变）徽章却翻成「波动中」并一直卡在那儿。
 * 因为 2.5s 的窗口在 6.2 Hz 下只装得下 15 帧，够不到 18 帧的门槛 ——
 * 和 POSE_DRIFT_DEG 那次踩的是同一类坑：**门槛卡在没有意义的维度上，
 * 于是变成死区而不是保护**。按时间判就没有这个问题：
 * 1500 ms（= 2500 ms 的 60%）在 6 帧/秒下是 9 帧，在 12 帧/秒下是 18 帧，都能过。
 */
const STABLE_FRACTION = 0.6
/**
 * 判为稳定的稳健标准差上限（分）。
 *
 * 两个夹具各跑 25 s 实测（`tmp/live/`，假摄像头，见 CLAUDE.md 的验证一节）：
 *
 * | 夹具 | ± 最小 | 中位 | 最大 | 判「稳定」的样本 |
 * |---|---|---|---|---|
 * | 静止（11.6 Hz） | 0 | 0 | 0.1 | 30 / 40 |
 * | 缓慢旋转 ±12°（9.3 Hz） | 0 | 1.4 | 5.1 | 0 / 40 |
 *
 * 静止那一档就是**检测器抖动的地板**（关键点逐帧抖，而引擎本身是确定性的），
 * 0.1 分离 3.0 很远，所以这道门不会误伤端住不动的人。
 *
 * ⚠️ 旋转那一档的 0/40 主要不是被这个数拦下的：它有 75% 的样本 ± ≤ 3.0，
 * 真正拦下它的是姿态判据（见 POSE_TRUST_DEG）。这是刻意的 ——
 * 转头时分数本来就不该被当成结论，而「稳定」这个徽章不该为它背书。
 */
const STABLE_SIGMA = 3.0
/** 输出 EMA 的系数。0.3 约合 3-4 个样本收敛，约 0.3s */
const EMA_ALPHA = 0.3
/**
 * 允许窗口内姿态相对**窗口起点**漂移的角度（度）。
 *
 * 判据是「漂移」而不是「绝对角度」—— 这是刻意的，两者会拦下完全不同的东西：
 *
 * - 绝对角度阈值（比如 6°）会误伤**端住不动**的人：一个稳定地歪着 10° 的人
 *   每帧都被判不合格，窗口被反复清空，界面就永远停在「正在测量…」，
 *   而用户看不出原因。这是死区，不是保护。
 * - 真正要防的是**缓慢转头**：中位数会被一帧帧合法的新样本带着从 80 走到 70，
 *   而每帧单独看都正常、离散度也小。
 *
 * 5° 的依据：实测同一张图 0°/10° 的综合分相差 1.1 分（80.1 vs 81.2），
 * 所以 5° 的漂移对应半分以内 —— 在文档反复强调的 ±5~8 分噪声之下，可以接受。
 *
 * 超过 20° 的帧由 quality.ts 拦掉（那边是拒判，会清空窗口），本模块不重复那个阈值。
 */
export const POSE_DRIFT_DEG = 5

/**
 * 读数可以标「稳定」的姿态上限（度）。
 *
 * 窗口内只要出现过超过这个角度的帧，即使离散度很小也不给稳定徽章：
 * 稳定只说明「没在动」，不说明「这个角度下的分数和正脸的可比」。
 * 取 8° 而不是 5°：它是给徽章用的，不是给窗口用的，容一点不会误导。
 */
export const POSE_TRUST_DEG = 8

/** 瞳距相对变化超过这个比例就认为人前后移动了，窗口作废 */
export const IPD_DRIFT = 0.05
/** 超过这个时间没有新样本就标记为陈旧 */
const STALE_MS = 800

export interface LiveSmoother {
  /**
   * 送入一帧。姿态相对窗口起点漂移过大、或瞳距明显变化时会**清空窗口**
   * 而不是仅仅不入栈 —— 留着旧样本正是让中位数被缓慢转头带着走的那个原因。
   *
   * 调用方**不要**把没过 quality.ts 的帧送进来：那些帧的姿态已经超出可信范围
   * （> 20°），应该由调用方自己 reset() 并给出解释。
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

  /** 按时间淘汰。push 和 read 都要做：只推不入的时候（画面里长时间没人脸）
   *  就没有 push 来触发淘汰，窗口会一直冻在那里，read 会永远报「陈旧」而不是
   *  自然地变成 null */
  function evict(now: number): void {
    const cutoff = now - MAX_AGE_MS
    while (samples.length > 0 && samples[0]!.t < cutoff) samples.shift()
  }

  function push(sample: LiveSample): boolean {
    if (!Number.isFinite(sample.score) || !Number.isFinite(sample.ipd)) return false
    // 拿不到姿态的帧不入窗：没有它就无法判断该不该作废窗口，
    // 而「无法判断」在一段会持续数秒的读数里等于放弃了唯一的运动检测
    if (!Number.isFinite(sample.maxPoseAngle)) return false

    const ref = samples[0]
    if (ref) {
      // 姿态漂移：见 POSE_DRIFT_DEG。判的是相对窗口起点的变化量，
      // 所以「稳定地偏着」不受影响，「慢慢转过去」会被抓住
      if (Math.abs(sample.maxPoseAngle - ref.maxPoseAngle) > POSE_DRIFT_DEG) reset()
      // 人前后移动了：尺度基准变了，旧样本和新样本不可比
      else if (ref.ipd > 0 && Math.abs(sample.ipd - ref.ipd) / ref.ipd > IPD_DRIFT) reset()
    }

    samples.push(sample)

    evict(sample.t)
    while (samples.length > MAX_SAMPLES) samples.shift()

    return true
  }

  function read(now: number): LiveReading | null {
    evict(now)
    if (samples.length < MIN_SAMPLES) return null

    const values = samples.map((s) => s.score).sort((a, b) => a - b)
    const center = medianSorted(values)
    const sigma = madSigma(values, center)

    // 中位数是阶梯状的，再过一道 EMA 让它连续
    smoothed = smoothed === null ? center : smoothed + EMA_ALPHA * (center - smoothed)

    let maxPoseAngle = 0
    for (const s of samples) maxPoseAngle = Math.max(maxPoseAngle, s.maxPoseAngle)

    const first = samples[0]!
    const last = samples[samples.length - 1]!
    // 窗口覆盖的时间跨度。见 STABLE_FRACTION 的说明：稳定要按时间判，
    // 按样本数判会把「设备慢」误报成「读数不稳」
    const span = last.t - first.t
    return {
      score: smoothed,
      uncertainty: sigma,
      n: samples.length,
      stable:
        samples.length >= MIN_SAMPLES &&
        span >= MAX_AGE_MS * STABLE_FRACTION &&
        sigma <= STABLE_SIGMA &&
        maxPoseAngle <= POSE_TRUST_DEG,
      maxPoseAngle,
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
