/**
 * 应用装配层：把界面事件接到 face / engine 上，并把结果画出来。
 *
 * 这里不写任何测量或推理逻辑 —— 那些在 face/ 与 engine/ 里，且都有各自
 * 独立的验证路径。这一层只负责串起来和把话说清楚。
 */

import { analyze, InsufficientLandmarksError, type AnalysisResult } from './engine/index.ts'
import { detectFaces, getActiveDelegate, loadLandmarker, type LoadStage } from './face/landmarker.ts'
import { checkQuality, type QualityReport } from './face/quality.ts'
import { prepareImage, type PreparedImage } from './image.ts'
import { drawOverlay } from './ui/overlay.ts'
import { renderReport } from './ui/report.ts'

const STAGE_TEXT: Record<LoadStage, string> = {
  wasm: '正在加载人脸检测引擎…',
  benchmark: '正在选择最快的推理方式…',
  model: '正在加载人脸模型…',
  ready: '',
}

interface Elements {
  drop: HTMLElement
  input: HTMLInputElement
  status: HTMLElement
  result: HTMLElement
  blocked: HTMLElement
  canvas: HTMLCanvasElement
  panel: HTMLElement
}

function need<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id)
  if (!el) throw new Error(`页面上缺少 #${id}`)
  return el as T
}

function showStatus(el: HTMLElement, text: string, kind: 'busy' | 'error' = 'busy'): void {
  el.hidden = false
  el.classList.toggle('is-error', kind === 'error')
  el.innerHTML =
    kind === 'busy' ? `<span class="status__spinner"></span>${text}` : text
}

function hide(el: HTMLElement): void {
  el.hidden = true
}

/** 把拒判 / 警告渲染成用户能看懂的一段话。 */
function renderQuality(container: HTMLElement, report: QualityReport): void {
  const rejects = report.issues.filter((i) => i.severity === 'reject')
  const warns = report.issues.filter((i) => i.severity === 'warn')
  if (rejects.length === 0 && warns.length === 0) {
    hide(container)
    return
  }

  container.hidden = false
  container.replaceChildren()

  if (rejects.length > 0) {
    const h = document.createElement('h2')
    h.textContent = '这张照片没法给出可靠的结果'
    const ul = document.createElement('ul')
    for (const issue of rejects) {
      const li = document.createElement('li')
      li.textContent = issue.message
      ul.append(li)
    }
    container.append(h, ul)
  }

  if (warns.length > 0) {
    const box = document.createElement('div')
    box.className = 'blocked__warn'
    const title = document.createElement('div')
    title.textContent = '另外提醒：'
    const ul = document.createElement('ul')
    for (const issue of warns) {
      const li = document.createElement('li')
      li.textContent = issue.message
      ul.append(li)
    }
    box.append(title, ul)
    container.append(box)
  }
}

function describeError(err: unknown): string {
  const message = err instanceof Error ? err.message : String(err)
  if (message.includes('could not be decoded') || message.includes('InvalidStateError')) {
    return '这个文件没法作为图片读取。请确认它是完整的 JPEG / PNG / WebP 图片。'
  }
  if (err instanceof InsufficientLandmarksError) {
    return '人脸关键点数据不完整，无法计算比例。'
  }
  if (message.includes('Failed to fetch') || message.includes('NetworkError')) {
    return '模型文件下载失败，请检查网络连接后重试。'
  }
  // 兜底：把原始信息也带上，方便反馈问题时定位
  return `处理失败：${message}`
}

export function mountApp(): void {
  const el: Elements = {
    drop: need('drop'),
    input: need<HTMLInputElement>('file'),
    status: need('status'),
    result: need('result'),
    blocked: need('blocked'),
    canvas: need<HTMLCanvasElement>('canvas'),
    panel: need('panel'),
  }

  /** 当前展示的图片，切换照片时释放，避免 ImageBitmap 泄漏 */
  let current: PreparedImage | null = null
  let busy = false

  function reset(): void {
    hide(el.result)
    hide(el.blocked)
    el.panel.replaceChildren()
    current?.bitmap.close()
    current = null
  }

  async function run(file: File): Promise<void> {
    if (busy) return
    if (!file.type.startsWith('image/')) {
      showStatus(el.status, '请选择图片文件（JPEG / PNG / WebP）。', 'error')
      return
    }

    busy = true
    reset()
    showStatus(el.status, '正在读取照片…')

    try {
      const prepared = await prepareImage(file)

      // 首次使用要下模型；这里也顺便拿到 'auto' 模式选后端所需的测速图
      const firstLoad = !getActiveDelegate()
      await loadLandmarker({
        delegate: 'auto',
        probe: prepared.bitmap,
        onStage: (stage) => {
          const text = STAGE_TEXT[stage]
          if (text) showStatus(el.status, text)
        },
        onDelegateChosen: (delegate) => {
          if (firstLoad) console.info(`[good-looking] 推理后端：${delegate}`)
        },
      })

      showStatus(el.status, '正在定位人脸…')
      const faces = detectFaces(prepared.bitmap)

      showStatus(el.status, '正在校验照片质量…')
      // 传位图是为了做模糊检测 —— 那一项要读原始像素，关键点里没有
      const quality = checkQuality(faces, prepared.bitmap)

      if (!quality.acceptable || !quality.face) {
        current = prepared
        renderQuality(el.blocked, quality)
        hide(el.status)
        return
      }

      // 还有警告也一并展示，但继续出分
      renderQuality(el.blocked, quality)

      let result: AnalysisResult
      try {
        result = analyze(quality.face.pixels)
      } catch (err) {
        current = prepared
        showStatus(el.status, describeError(err), 'error')
        return
      }

      current = prepared
      drawOverlay(el.canvas, prepared.bitmap, quality.face, result)
      renderReport(el.panel, result)

      el.result.hidden = false
      hide(el.status)
    } catch (err) {
      console.error('[good-looking]', err)
      showStatus(el.status, describeError(err), 'error')
    } finally {
      busy = false
    }
  }

  el.input.addEventListener('change', () => {
    const file = el.input.files?.[0]
    if (file) void run(file)
    // 清空 value，否则连续选同一张图不会再触发 change
    el.input.value = ''
  })

  // 拖放
  el.drop.addEventListener('dragover', (e) => {
    e.preventDefault()
    el.drop.classList.add('is-over')
  })
  el.drop.addEventListener('dragleave', () => el.drop.classList.remove('is-over'))
  el.drop.addEventListener('drop', (e) => {
    e.preventDefault()
    el.drop.classList.remove('is-over')
    const file = e.dataTransfer?.files?.[0]
    if (file) void run(file)
  })
}
