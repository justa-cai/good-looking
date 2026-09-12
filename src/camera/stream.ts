/**
 * 摄像头取流与生命周期。
 *
 * 只负责「拿到一路可用的视频流，并且能干净地放掉」这一件事：不碰关键点、
 * 不碰分数、不碰界面。这样它和 face/ 一样是独立可测的一层。
 *
 * 这里的复杂度几乎全部来自**失败路径**。摄像头是本项目第一种会持续失败、
 * 且失败原因用户自己看不见的输入：权限被拒、设备被别的程序占着、USB 拔了、
 * 页面跑在非安全上下文里 —— 每种的处置都不一样，而浏览器的报错都是同一句
 * 没有信息量的 `NotAllowedError` / `NotReadableError`。所以下面把它们分开映射，
 * 并且尽量给出「用户下一步该做什么」而不是把原文抛出去。
 */

/**
 * 摄像头启动失败的原因。
 *
 * 分开列举是为了让界面能给出**不同的处置建议** —— 权限被拒要让用户去改站点设置
 * （再点一次没用），没有设备要让用户插一个，被占用要让用户关掉别的程序。
 * 归成一个「摄像头打不开」等于什么都没说。
 */
export type CameraFailure =
  /** 非安全上下文，`navigator.mediaDevices` 根本不存在 */
  | 'insecure'
  /** 权限被拒（站点级） */
  | 'denied'
  /** 本机没有可用摄像头（含被系统级策略禁用） */
  | 'no-camera'
  /** 设备存在但打不开：被别的程序占用、驱动故障、系统隐私开关关着 */
  | 'unavailable'
  /** 约束无法满足 */
  | 'constraints'
  /** 拿不到元数据（流建立了但画面一直不来） */
  | 'timeout'
  | 'unknown'

export class CameraError extends Error {
  readonly failure: CameraFailure
  /**
   * 浏览器抛出的**原始**错误。
   *
   * 必须留着。我们映射出来的文案是给用户看的，而这里才是排查时唯一能看的东西：
   * 丢了它，出问题时控制台里只剩一句我们自己编的话，连到底是 `NotFoundError`
   * 还是 `NotAllowedError` 都分不出来 —— 而这两种的处置完全不同。
   */
  readonly original: unknown

  constructor(failure: CameraFailure, message: string, original?: unknown) {
    super(message)
    this.name = 'CameraError'
    this.failure = failure
    this.original = original
  }
}

/**
 * 请求的约束。
 *
 * 全部用 `ideal` 而不是 `exact`：`exact` 会让一台只支持 640×480 的笔记本摄像头
 * 直接 `OverconstrainedError` 失败，而 640×480 完全够用（CLAUDE.md 记了瞳距
 * 阈值在这个分辨率下的表现）。这里的原则是「尽量给好的，不满足就用现有的」。
 *
 * 要 1280×720 是为了让脸占到足够的像素 —— 关键点精度取决于脸的像素数，
 * 而 `minIpd` 这类阈值是绝对像素值。
 */
const CONSTRAINTS: MediaStreamConstraints = {
  video: {
    // 自拍场景，用前置摄像头。桌面浏览器通常只有一个，无差别
    facingMode: 'user',
    width: { ideal: 1280 },
    height: { ideal: 720 },
  },
  // 只要视频。带上音频会让权限弹窗多问一项，用户多一分犹豫，而我们不用它
  audio: false,
}

/** 等元数据的上限。超时就报错，不无限挂着 */
const METADATA_TIMEOUT_MS = 10_000

export interface CameraStartOptions {
  /**
   * track 意外结束时的回调（USB 被拔、被别的程序抢走、系统撤销授权）。
   *
   * **必须处理**：track 结束后 `<video>` 会停在最后一帧画面不动，看起来一切正常，
   * 而帧循环还在以 10Hz 跑着同一张静止画面、反复把同一组关键点喂进平滑窗口 ——
   * 界面会用一个漂亮的小不确定度显示一个完全僵死的分数。
   * 只回调一次（超时 / 手动 stop 都不会触发）。
   */
  onEnded?: () => void
  /** 等待 `videoWidth > 0` 的超时（毫秒） */
  timeoutMs?: number
}

export interface CameraSession {
  readonly stream: MediaStream
  readonly track: MediaStreamTrack
  /** 画面像素宽度。返回时保证 > 0 */
  readonly width: number
  /** 画面像素高度。返回时保证 > 0 */
  readonly height: number
  /**
   * 浏览器实际采用的设置。`width/height` 是**请求**值，这里才是真实值 ——
   * 两者不一致是常态（摄像头不支持 720p，或系统在低光下自动降分辨率）。
   * 关键点坐标基于真实尺寸，所以拿它来排查定位问题。
   */
  readonly settings: MediaTrackSettings
  /** 放掉摄像头。幂等，可重复调用 */
  stop(): void
}

/** `navigator.mediaDevices` 在非安全上下文里是 undefined，不是「调用后报错」。 */
export function isCameraSupported(): boolean {
  return typeof navigator !== 'undefined' && !!navigator.mediaDevices?.getUserMedia
}

/**
 * 打开摄像头并把流接到 `video` 上。
 *
 * 返回时保证 `video.videoWidth > 0` —— 关键点坐标要乘以视频尺寸，
 * 元数据没到时那个值是 0，所有像素坐标都会变成 0，最后只会得到一句
 * 「瞳距无效」，和真实原因（画面还没来）对不上。
 */
export async function startCamera(
  video: HTMLVideoElement,
  opts: CameraStartOptions = {},
): Promise<CameraSession> {
  const { onEnded, timeoutMs = METADATA_TIMEOUT_MS } = opts

  // 显式判断而不是靠 catch。局域网 IP + http 访问时 `navigator.mediaDevices`
  // 是 undefined，`getUserMedia` 会抛 TypeError（"cannot read property of undefined"），
  // 报出来的东西和真实原因毫无关系。和 CLAUDE.md 里 Cache API 那条限制同类。
  if (!isCameraSupported()) {
    throw new CameraError(
      'insecure',
      '当前页面不是安全上下文，浏览器不允许访问摄像头。请用 https 或 localhost 打开。',
    )
  }

  let stream: MediaStream
  try {
    stream = await navigator.mediaDevices.getUserMedia(CONSTRAINTS)
  } catch (err) {
    const mapped = await mapGetUserMediaError(err)
    // 摄像头是**唯一**一种「失败原因在用户那台机器上、而我们这边什么都看不到」的
    // 输入：报错只有一两个没有信息量的名字，本地又复现不了。所以失败时把能观察到的
    // 东西一次性打出来 —— 用户贴回来就能分清是系统权限、设备被占、还是浏览器
    // 沙箱里根本没有设备，而不是靠猜。
    console.error(
      `[good-looking] 打开摄像头失败（${mapped.failure}）：${mapped.message}\n环境诊断：\n${await describeCameraSupport()}`,
      err,
    )
    throw mapped
  }

  const track = stream.getVideoTracks()[0]
  if (!track) {
    // 理论上不会发生（audio:false，拿到流就必有 video track），
    // 但真发生时上面的 stop() 才是唯一能关掉设备的地方，不能漏
    for (const t of stream.getTracks()) t.stop()
    throw new CameraError('unavailable', '浏览器返回的视频流里没有视频轨道。')
  }

  let stopped = false
  let endedFired = false

  const onTrackEnded = (): void => {
    if (stopped || endedFired) return
    endedFired = true
    onEnded?.()
  }
  track.addEventListener('ended', onTrackEnded)

  const stop = (): void => {
    if (stopped) return
    stopped = true
    track.removeEventListener('ended', onTrackEnded)
    // 真正关掉设备。只置 srcObject = null 的话摄像头指示灯还亮着 ——
    // 对一个把「照片不离开设备」当核心承诺的项目，这是不能接受的
    for (const t of stream.getTracks()) t.stop()
    video.pause()
    video.srcObject = null
  }

  // iOS Safari 不加 playsinline 会强制全屏播放；muted 是自动播放的前提
  video.playsInline = true
  video.muted = true
  video.srcObject = stream

  try {
    const { width, height } = await waitForMetadata(video, timeoutMs, () => endedFired)
    await video.play()
    return {
      stream,
      track,
      width,
      height,
      settings: track.getSettings(),
      stop,
    }
  } catch (err) {
    stop()
    throw err
  }
}

/**
 * 等 `videoWidth > 0`。
 *
 * 同时盯着三种提前失败：track 结束（拔了 / 被抢）、超时、以及 `loadedmetadata`
 * 到了但尺寸仍是 0（畸形流）。哪一种都不能让它一直挂着。
 */
function waitForMetadata(
  video: HTMLVideoElement,
  timeoutMs: number,
  isEnded: () => boolean,
): Promise<{ width: number; height: number }> {
  if (video.videoWidth > 0 && video.videoHeight > 0) {
    return Promise.resolve({ width: video.videoWidth, height: video.videoHeight })
  }

  return new Promise((resolve, reject) => {
    let timer: number | undefined

    const cleanup = (): void => {
      video.removeEventListener('loadedmetadata', onMeta)
      if (timer !== undefined) clearTimeout(timer)
    }
    const fail = (err: Error): void => {
      cleanup()
      reject(err)
    }
    const onMeta = (): void => {
      if (isEnded()) {
        fail(new CameraError('unavailable', '摄像头在读到时序信息前就断开了。'))
        return
      }
      // 有的实现会在 loadedmetadata 时还报 0，等下一次 canplay 更稳，
      // 但那种情况极少；这里报错比静默返回 0 好，后者会一路传到「瞳距无效」
      if (video.videoWidth <= 0 || video.videoHeight <= 0) {
        fail(new CameraError('unavailable', '摄像头给出的画面尺寸是 0，无法使用。'))
        return
      }
      cleanup()
      resolve({ width: video.videoWidth, height: video.videoHeight })
    }

    video.addEventListener('loadedmetadata', onMeta)
    timer = window.setTimeout(() => {
      if (isEnded()) {
        fail(new CameraError('unavailable', '摄像头在读到时序信息前就断开了。'))
        return
      }
      fail(
        new CameraError(
          'timeout',
          `等了 ${Math.round(timeoutMs / 1000)} 秒还没拿到摄像头画面。请检查是不是有别的程序正在用它。`,
        ),
      )
    }, timeoutMs)
  })
}

/**
 * 把 `getUserMedia` 的报错翻译成用户能处置的话。
 *
 * 关键点：`NotAllowedError` **同时**表示「用户拒绝了」和「系统层面禁用了摄像头」
 * （Windows 的隐私设置、macOS 的屏幕使用时间限制都会走到这里），而这两种的
 * 处置完全不同。用 `enumerateDevices()` 来区分 —— 它不需要权限就能列出设备种类，
 * 一个 videoinput 都没有就说明这台机器上浏览器看不到任何摄像头。
 *
 * ⚠️ 措辞上的纪律：**只说自己确知的事**。我们能确知的是「浏览器没能打开任何摄像头」，
 * 不确知用户桌上到底摆没摆一个摄像头 —— 系统权限、别的程序独占、浏览器跑在沙箱里
 * 都会让我们看不到它。所以文案要把这些可能性摆出来，而不是断言「你没有摄像头」。
 */
async function mapGetUserMediaError(err: unknown): Promise<CameraError> {
  if (err instanceof CameraError) return err

  const name = err instanceof DOMException ? err.name : ''
  const detail = err instanceof Error ? err.message : String(err)

  switch (name) {
    case 'NotAllowedError': {
      if (!(await hasVideoInput())) {
        return new CameraError(
          'no-camera',
          '浏览器看不到任何可用的摄像头。最可能的原因是系统还没允许浏览器使用摄像头（macOS 的「隐私与安全性 → 摄像头」、Windows 的「隐私和安全性 → 相机」），也可能是摄像头正被别的程序独占。确认之后刷新页面再试。控制台里有一条环境诊断，可据此进一步定位。',
          err,
        )
      }
      const state = await cameraPermissionState()
      if (state === 'denied') {
        return new CameraError(
          'denied',
          '摄像头权限被拒绝了。这个设置会被浏览器记住，再点一次「开始」不会重新弹窗 —— 请点地址栏左侧的图标，把摄像头改回「允许」，然后刷新页面。',
          err,
        )
      }
      return new CameraError(
        'denied',
        '没能拿到摄像头权限。可能是刚才的弹窗被关掉了，也可能是系统层面禁用了浏览器使用摄像头。确认之后请再试一次。',
        err,
      )
    }
    case 'NotFoundError':
    case 'DevicesNotFoundError':
      return new CameraError(
        'no-camera',
        '浏览器没能找到任何摄像头。常见原因是系统还没允许浏览器使用摄像头（macOS 的「隐私与安全性 → 摄像头」、Windows 的「隐私和安全性 → 相机」），或者摄像头正被别的程序独占着。确认之后刷新页面再试。控制台里有一条环境诊断，可据此进一步定位。',
        err,
      )
    case 'NotReadableError':
    case 'TrackStartError':
      return new CameraError(
        'unavailable',
        '摄像头打不开，多半是正被别的程序（视频会议、录屏软件）占用。关掉它们再试。',
        err,
      )
    case 'OverconstrainedError':
    case 'ConstraintNotSatisfiedError':
      return new CameraError(
        'constraints',
        `摄像头满足不了所需的分辨率（${detail}）。请换一个摄像头，或改用在其他设备上打开。`,
        err,
      )
    case 'SecurityError':
      return new CameraError(
        'insecure',
        '浏览器出于安全策略拒绝了摄像头访问。请用 https 或 localhost 打开。',
        err,
      )
    case 'AbortError':
      return new CameraError('unavailable', '摄像头启动被中断，请重试。', err)
    default:
      return new CameraError('unknown', `打开摄像头失败：${detail}`, err)
  }
}

/** 有没有 videoinput。`enumerateDevices()` 不需要摄像头权限 */
async function hasVideoInput(): Promise<boolean> {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices()
    return devices.some((d) => d.kind === 'videoinput')
  } catch {
    // 查不到就当有 —— 宁可提示「可能没摄像头」，也不要因为查询本身失败
    // 就把「用户拒绝授权」误报成「本机没摄像头」，那是两个完全不同的处置
    return true
  }
}

/**
 * 摄像头相关的环境诊断，**只在启动失败时采集并打印**。
 *
 * 为什么值得写这么一段：摄像头失败的现场在**用户那台机器上**，我们这边
 * 什么信号都收不到，而 `getUserMedia` 只给一两个没有信息量的错误名 ——
 * 本地又几乎不可能复现（本机就压根没有摄像头，一直以来只能用假摄像头测）。
 * 把能观察到的东西一次性打全，用户贴回来就能定位，不用来回猜。
 */
export async function describeCameraSupport(): Promise<string> {
  const lines: string[] = []
  const put = (k: string, v: unknown): void => {
    lines.push(`  ${k}: ${String(v)}`)
  }

  put('isSecureContext', typeof isSecureContext === 'boolean' ? isSecureContext : '未知')
  // 带上 UA 就是为了知道「是哪台机器上的哪个浏览器」—— 同一句报错在
  // macOS 系统权限、Windows 隐私开关、Linux 的 snap 沙箱下含义完全不同
  put('userAgent', typeof navigator !== 'undefined' ? navigator.userAgent : '未知')
  put('mediaDevices 存在', typeof navigator !== 'undefined' && !!navigator.mediaDevices)
  put('getUserMedia 可用', isCameraSupported())

  if (navigator?.mediaDevices?.enumerateDevices) {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices()
      const counts = new Map<string, number>()
      for (const d of devices) counts.set(d.kind, (counts.get(d.kind) ?? 0) + 1)
      put(
        '设备计数',
        [...counts].map(([kind, n]) => `${kind}×${n}`).join('、') || '一个都没有',
      )
      const video = devices.filter((d) => d.kind === 'videoinput')
      put(
        '摄像头',
        video.map((d) => d.label || '(无标签)').join(' | ') ||
          '一个都没有（系统权限、设备被独占、或浏览器跑在沙箱里都会是这样）',
      )
    } catch (err) {
      put('enumerateDevices', `失败：${err instanceof Error ? err.message : String(err)}`)
    }
  }

  try {
    // 'camera' 在当前的 TS lib.dom 里还不在 PermissionName 联合类型里
    const status = await navigator.permissions.query({ name: 'camera' as PermissionName })
    put('站点权限状态', status.state)
  } catch (err) {
    put('站点权限状态', `查不到（${err instanceof Error ? err.name : String(err)}）`)
  }

  return lines.join('\n')
}

/**
 * 站点级的摄像头权限状态。
 *
 * `'denied'` 是**粘住**的，重新调用 `getUserMedia` 不会再弹窗，所以这个区分
 * 决定了要不要让用户去翻站点设置。Safari 至今不支持 `camera` 这个 PermissionName，
 * 查不到就返回 null，走那句覆盖面更广的文案。
 */
async function cameraPermissionState(): Promise<PermissionState | null> {
  try {
    // 'camera' 在当前的 TS lib.dom 里还不在 PermissionName 联合类型里
    const status = await navigator.permissions.query({
      name: 'camera' as PermissionName,
    })
    return status.state
  } catch {
    return null
  }
}
