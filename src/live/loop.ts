/**
 * 实时帧循环。
 *
 * 只做调度：什么时候跑一帧、跑崩了怎么办、页面切到后台怎么办。
 * 每帧真正做什么由调用方给的回调决定，所以这一层不认识关键点也不认识分数。
 *
 * 用 `requestVideoFrameCallback`（rVFC）而不是 `requestAnimationFrame` 驱动：
 * rVFC 只在**视频真有新帧**时触发，而且回调里带的 `now` 是那一帧的呈现时间，
 * 正好是 MediaPipe 要的严格递增时间戳。rAF 在 60Hz 屏幕上每秒会叫 60 次，
 * 而摄像头只有 30fps，一半的调用会白跑一遍检测。Firefox 早期版本没有 rVFC，
 * 所以保留 rAF 兜底。
 */

/** 两次检测之间的最小间隔（毫秒）。约 12Hz —— 检测本身要 30-45ms，这是上限不是节流 */
const MIN_INTERVAL_MS = 80

/**
 * 连续失败多少次就放弃。
 *
 * MediaPipe 的时间戳违例会让 C++ graph **永久卡死**，之后每一帧都抛同样的错 ——
 * 那种情况下重试一万次也不会有一次成功，只会让界面每 80ms 闪一下错误。
 * 所以宁可停下来大声报错，由上层决定重建实例。
 * 取 3 而不是 1：偶发的一帧失败（画布被 resize、关键点暂时不足）是正常的。
 */
const MAX_CONSECUTIVE_FAILURES = 3

export interface FrameLoopCallbacks {
  /**
   * 每一帧。抛出的任何错误都算这一帧失败，由循环计数。
   *
   * @param now 这一帧的时间戳，与 `performance.now()` 同源。
   *   **只能用这个值**去调 `detectFacesFromVideo` —— rVFC 的 `metadata.mediaTime`
   *   是另一条从 0 开始的时间线，混用会让 MediaPipe 的时间戳守卫失去意义。
   * @param deltaMs 距离上一帧的间隔。首帧为 0
   */
  onFrame(now: number, deltaMs: number): void
  /** 单帧失败但还没到放弃阈值。用于界面上的瞬时提示 */
  onError?(err: unknown): void
  /** 连续失败达到阈值，循环已经停下（不会再有 onFrame） */
  onFatal(err: unknown): void
  /** 页面切到后台 / 切回来。切走时循环不再调度，切回时从零恢复 */
  onVisibilityChange?(hidden: boolean): void
}

export interface FrameLoop {
  /** 幂等。重复调用不会跑起第二个循环 */
  start(): void
  stop(): void
  /** 是否正在跑。页面隐藏期间是 false（暂停），但 `stop()` 还没被调用 */
  isRunning(): boolean
}

export function createFrameLoop(
  video: HTMLVideoElement,
  callbacks: FrameLoopCallbacks,
): FrameLoop {
  const useRvfc = typeof video.requestVideoFrameCallback === 'function'

  let running = false
  /** 页面是否处于隐藏状态。隐藏时暂停调度，但不算 stop */
  let hidden = false
  let failures = 0
  let lastTs = Number.NEGATIVE_INFINITY
  let rvfcHandle: number | null = null
  let rafHandle: number | null = null

  function cancelPending(): void {
    if (rvfcHandle !== null && typeof video.cancelVideoFrameCallback === 'function') {
      video.cancelVideoFrameCallback(rvfcHandle)
    }
    if (rafHandle !== null) cancelAnimationFrame(rafHandle)
    rvfcHandle = null
    rafHandle = null
  }

  /**
   * 登记下一次回调。
   *
   * 关键：这**只在上一帧彻底跑完之后**才被调用（见 onFrame 的 finally），
   * 所以检测耗时期间的视频帧会被自然跳过，不会堆积回调、不会越跑越滞后。
   */
  function schedule(): void {
    if (!running || hidden) return
    if (useRvfc) {
      rvfcHandle = video.requestVideoFrameCallback((now) => {
        rvfcHandle = null
        try {
          tick(now)
        } finally {
          schedule()
        }
      })
    } else {
      rafHandle = requestAnimationFrame((now) => {
        rafHandle = null
        try {
          tick(now)
        } finally {
          schedule()
        }
      })
    }
  }

  function tick(now: number): void {
    // 节流。rVFC 被跳过时不会消耗这一帧的时间戳（我们根本没把它交给检测器），
    // 所以跳过是安全的
    if (now - lastTs < MIN_INTERVAL_MS) return

    const deltaMs = lastTs === Number.NEGATIVE_INFINITY ? 0 : now - lastTs
    lastTs = now

    try {
      callbacks.onFrame(now, deltaMs)
      failures = 0
    } catch (err) {
      failures++
      if (failures >= MAX_CONSECUTIVE_FAILURES) {
        stop()
        callbacks.onFatal(err)
      } else {
        callbacks.onError?.(err)
      }
    }
  }

  function onVisibility(): void {
    const nowHidden = document.hidden
    if (nowHidden === hidden) return
    hidden = nowHidden

    if (hidden) {
      // rVFC/rAF 在后台本来就会停，但**摄像头还开着**，回来时流的画面已经跳了
      // 好几秒。主动取消挂起的回调，别指望浏览器
      cancelPending()
      callbacks.onVisibilityChange?.(true)
    } else {
      // 恢复后的第一帧不该被 MIN_INTERVAL 挡住：lastTs 是很久以前的值，
      // 本来就会通过。这里只是把它显式化，避免以后改 MIN_INTERVAL 时踩到
      lastTs = Number.NEGATIVE_INFINITY
      callbacks.onVisibilityChange?.(false)
      schedule()
    }
  }

  function start(): void {
    if (running) return
    running = true
    hidden = document.hidden
    failures = 0
    lastTs = Number.NEGATIVE_INFINITY
    document.addEventListener('visibilitychange', onVisibility)
    // 页面本来就是隐藏的话（后台标签里点的开始），onVisibility 已经把它算进
    // hidden，schedule 会直接返回，等切回来再跑
    schedule()
  }

  function stop(): void {
    if (!running) return
    running = false
    document.removeEventListener('visibilitychange', onVisibility)
    cancelPending()
  }

  return { start, stop, isRunning: () => running }
}

/**
 * 循环用的是 rVFC 还是 rAF。给界面的调试读数用 ——
 * 两种模式的节奏和滞后不一样，出问题时这是第一个要看的东西。
 */
export function frameDriver(video: HTMLVideoElement): 'rvfc' | 'raf' {
  return typeof video.requestVideoFrameCallback === 'function' ? 'rvfc' : 'raf'
}
