/**
 * 实时取景面板的界面层。
 *
 * 只负责画和写：拿到 `LiveFrame` 就往 DOM 上刷，自己不判断分数该怎么算。
 * 判定逻辑在 `live/session.ts`，两者分开是为了让判定能被探针脚本直接调用。
 *
 * 两条无障碍上的硬要求：
 *
 * 1. **读数区不能是 live region。** 它 12Hz 更新一次，放进 `aria-live` 会把
 *    读屏软件刷爆。所以读数用 `aria-live="off"`，只有取景提示那一行走
 *    `polite` —— 它的文案带 400ms 停留，本来就是低频的。
 * 2. **数字一律取整。** CLAUDE.md 明确说不要把个位数的分差当成有意义的差异：
 *    显示「78.4」再让它跳到「77.1」直接违背这一点。整数 + 显式的 ±不确定度。
 */

import type { LiveFrame } from '../live/session.ts'

export interface LiveActions {
  onStart(): void
  onStop(): void
  /** 冻结当前帧，走完整流程 */
  onFreeze(): void
  /** 从冻结结果回到实时 */
  onBackToLive(): void
}

export type LiveUiState = 'idle' | 'starting' | 'running' | 'frozen' | 'working'

export interface LiveView {
  readonly video: HTMLVideoElement
  readonly canvas: HTMLCanvasElement
  /** 每帧刷新。传 null 表示回到未开始状态 */
  update(frame: LiveFrame | null): void
  setState(state: LiveUiState): void
  /** 加载阶段的状态文字。null 清掉 */
  setStatus(text: string | null, kind?: 'busy' | 'error'): void
  /** 长期性的错误：一直留在面板里，直到下一次成功 */
  setError(message: string | null): void
  /**
   * 帧循环用的是什么驱动（rVFC / rAF）。
   *
   * 只是调试读数的一部分，但很值得显示：两种模式的节奏和滞后不一样，
   * 用户报「卡顿」时这是第一个要看的东西，而在开发机上复现不了
   * （CLAUDE.md 记着本机 GPU 性能数据不可信这条限制）。
   */
  setDriver(driver: 'rvfc' | 'raf'): void
}

/** 只改文本，值没变就不碰 DOM —— 12Hz 下这个判断省掉大量无谓的重排 */
function setText(el: HTMLElement, text: string): void {
  if (el.textContent !== text) el.textContent = text
}

function setClass(el: HTMLElement, name: string, on: boolean): void {
  el.classList.toggle(name, on)
}

export function createLiveView(container: HTMLElement, actions: LiveActions): LiveView {
  container.replaceChildren()

  const stage = document.createElement('div')
  stage.className = 'live__stage'

  /**
   * `<video>` 只是取帧的源，不用来显示（画面由 canvas 画，这样保证
   * 「看到的那一帧就是被测量的那一帧」）。
   *
   * ⚠️ 不能 `display:none`：部分浏览器会因此停止解码，rVFC 也就不再触发。
   * 用「1px + 透明 + 不可交互」把它藏在画面外。
   */
  const video = document.createElement('video')
  video.className = 'live__source'
  video.playsInline = true
  video.muted = true
  video.setAttribute('aria-hidden', 'true')

  const canvas = document.createElement('canvas')
  canvas.className = 'live__canvas'

  /** 取景提示：唯一允许是 live region 的地方 */
  const hint = document.createElement('p')
  hint.className = 'live__hint'
  hint.setAttribute('aria-live', 'polite')

  const placeholder = document.createElement('p')
  placeholder.className = 'live__placeholder'
  placeholder.textContent = '点下面的「开始」打开摄像头。画面只在这台设备上处理。'

  stage.append(video, canvas, hint, placeholder)

  const panel = document.createElement('div')
  panel.className = 'live__panel'

  const scoreBox = document.createElement('div')
  scoreBox.className = 'live__score'
  scoreBox.setAttribute('aria-live', 'off')
  const scoreNum = document.createElement('span')
  scoreNum.className = 'live__num'
  const scoreUnit = document.createElement('span')
  scoreUnit.className = 'live__unit'
  scoreUnit.textContent = '/100'
  const scoreUnc = document.createElement('span')
  scoreUnc.className = 'live__unc'
  const badge = document.createElement('span')
  badge.className = 'live__badge'
  scoreBox.append(scoreNum, scoreUnit, scoreUnc, badge)

  const note = document.createElement('p')
  note.className = 'live__note'
  note.setAttribute('aria-live', 'off')

  const debug = document.createElement('p')
  debug.className = 'live__debug'
  debug.setAttribute('aria-live', 'off')

  const status = document.createElement('p')
  status.className = 'live__status'
  status.hidden = true

  const error = document.createElement('p')
  error.className = 'live__error'
  error.hidden = true
  // 摄像头打不开这种要用户动手的错误值得播报一次
  error.setAttribute('role', 'status')

  const actionsBox = document.createElement('div')
  actionsBox.className = 'live__actions'

  const btnStart = button('开始', 'live__btn live__btn--primary', () => actions.onStart())
  const btnStop = button('停止', 'live__btn', () => actions.onStop())
  const btnFreeze = button('冻结这一帧', 'live__btn live__btn--primary', () => actions.onFreeze())
  const btnBack = button('返回实时', 'live__btn', () => actions.onBackToLive())
  actionsBox.append(btnStart, btnStop, btnFreeze, btnBack)

  const explainer = document.createElement('p')
  explainer.className = 'live__explainer'
  explainer.innerHTML =
    '实时分数是<strong>最近 2.5 秒的中位数</strong>，不是单帧值 —— 单张照片本身就有 ±5~8 分的噪声，' +
    '逐帧的原值跳得没法看。下面的 ± 是这段窗口的稳健标准差，' +
    '徽章表示这段读数稳不稳定。<strong>冻结之后的分数可能和这里差几分</strong>，' +
    '因为实时用的是逐帧跟踪、冻结走的是单张照片那条路，两者对同一帧的关键点本来就不完全一样 ——' +
    '以冻结那一次的结果为准。也<strong>不要拿它和上传照片算出的分</strong>对比。'

  panel.append(scoreBox, note, debug, status, error, actionsBox, explainer)
  container.append(stage, panel)

  let lastScore = ''
  let lastBadge = ''
  let lastNote = ''
  let lastDebug = ''
  let lastHint = ''
  /** 帧循环的驱动方式，setDriver 之前是空串 */
  let driver = ''

  function button(text: string, cls: string, onClick: () => void): HTMLButtonElement {
    const b = document.createElement('button')
    b.type = 'button'
    b.className = cls
    b.textContent = text
    b.addEventListener('click', onClick)
    return b
  }

  function setState(state: LiveUiState): void {
    btnStart.hidden = state !== 'idle'
    btnStop.hidden = state === 'idle' || state === 'starting'
    btnFreeze.hidden = state !== 'running'
    btnBack.hidden = state !== 'frozen' && state !== 'working'
    btnStop.disabled = state === 'working'
    btnBack.disabled = state === 'working'
    btnFreeze.disabled = state === 'working'
    // 冻结/工作中不该再显示实时读数：结构上同一时间只能有一个分数
    const showReadout = state === 'running' || state === 'starting'
    scoreBox.hidden = !showReadout
    note.hidden = !showReadout
    debug.hidden = !showReadout
    hint.hidden = !showReadout
    placeholder.hidden = state !== 'idle'
    stage.classList.toggle('is-live', showReadout)
    // 用 class 而不是 hidden：`.live__canvas { display: block }` 会盖掉
    // UA 对 [hidden] 的 display:none，这是最容易踩的那个坑
    canvas.classList.toggle('is-ready', showReadout)
    if (state === 'idle') {
      setText(scoreNum, '—')
      setText(scoreUnc, '')
      setText(badge, '')
      setText(note, '')
      setText(debug, '')
      setText(hint, '')
      lastScore = lastBadge = lastNote = lastDebug = lastHint = ''
    }
  }

  function update(frame: LiveFrame | null): void {
    if (!frame) {
      setText(hint, '')
      return
    }

    const reading = frame.reading

    // ---- 分数与不确定度 ----
    const scoreText = reading ? String(Math.round(reading.score)) : '—'
    if (scoreText !== lastScore) {
      setText(scoreNum, scoreText)
      lastScore = scoreText
    }
    setText(
      scoreUnc,
      reading ? `±${reading.uncertainty.toFixed(1)}` : '',
    )
    // 陈旧时把数字变暗，而不是继续自信地亮着
    scoreBox.classList.toggle('is-stale', reading?.stale === true)

    // ---- 稳定徽章 ----
    const badgeText = !reading
      ? '正在测量'
      : reading.stale
        ? '读数变旧'
        : reading.stable
          ? '稳定'
          : '波动中'
    if (badgeText !== lastBadge) {
      setText(badge, badgeText)
      badge.className = 'live__badge'
      badge.classList.add(
        !reading
          ? 'is-measuring'
          : reading.stale
            ? 'is-stale'
            : reading.stable
              ? 'is-stable'
              : 'is-noisy',
      )
      lastBadge = badgeText
    }

    // ---- 说明行：为什么会这样 ----
    const pose = frame.pose
    const poseText = pose
      ? `左右 ${pose.yaw.toFixed(0)}° · 俯仰 ${pose.pitch.toFixed(0)}° · 倾斜 ${pose.roll.toFixed(0)}°`
      : '姿态不可用'
    const parts = [poseText, `瞳距 ${Math.round(frame.ipd)}px`]
    if (reading) parts.push(`窗口 ${reading.n} 帧`)
    if (frame.rawScore !== null) parts.push(`本帧原始分 ${frame.rawScore.toFixed(1)}`)
    const noteText = parts.join(' · ')
    if (noteText !== lastNote) {
      setText(note, noteText)
      lastNote = noteText
    }

    // ---- 调试读数：把「本机测不出有效结论」的东西交给用户设备去量 ----
    const sharp =
      frame.sharpness === null
        ? '清晰度未测'
        : `清晰度 ${frame.sharpness.toFixed(1)}（${
            frame.sharpnessState === 'bad'
              ? '偏糊'
              : frame.sharpnessState === 'soft'
                ? '略软'
                : '正常'
          }）`
    const debugText = `${frame.fps.toFixed(1)} Hz · ${driver} · ${sharp}`
    if (debugText !== lastDebug) {
      setText(debug, debugText)
      lastDebug = debugText
    }

    const hintText = frame.guidance?.text ?? ''
    if (hintText !== lastHint) {
      setText(hint, hintText)
      setClass(hint, 'is-warn', frame.guidance?.tone === 'warn')
      lastHint = hintText
    }
  }

  function setStatus(text: string | null, kind: 'busy' | 'error' = 'busy'): void {
    if (!text) {
      status.hidden = true
      status.replaceChildren()
      return
    }
    status.hidden = false
    setClass(status, 'is-error', kind === 'error')
    if (kind === 'busy') {
      const spinner = document.createElement('span')
      spinner.className = 'status__spinner'
      status.replaceChildren(spinner, document.createTextNode(text))
    } else {
      setText(status, text)
    }
  }

  function setError(message: string | null): void {
    error.hidden = !message
    setText(error, message ?? '')
  }

  function setDriver(next: 'rvfc' | 'raf'): void {
    driver = next
  }

  setState('idle')

  return { video, canvas, update, setState, setStatus, setError, setDriver }
}
