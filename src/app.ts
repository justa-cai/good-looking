/**
 * 应用装配层：把界面事件接到 face / engine 上，并把结果画出来。
 *
 * 这里不写任何测量或推理逻辑 —— 那些在 face/ 与 engine/ 里，且都有各自
 * 独立的验证路径。这一层只负责串起来和把话说清楚。
 */

import { analyze, InsufficientLandmarksError, type AnalysisResult } from './engine/index.ts'
import { alignFace } from './face/align.ts'
import { detectFaces, getActiveDelegate, loadLandmarker, type LoadStage } from './face/landmarker.ts'
import { checkQuality, type QualityReport } from './face/quality.ts'
import { prepareImage, type PreparedImage } from './image.ts'
import { listCachedModels } from './model/cache.ts'
import { renderModelCard, type ModelCardState } from './ui/model_card.ts'
import { drawOverlay } from './ui/overlay.ts'
import { renderReport } from './ui/report.ts'

/**
 * 路线 B 的推理模块**按需加载**，不静态 import。
 *
 * 理由：`onnxruntime-web` 打出来是 403 kB（gzip 109 kB），而这个包只有
 * 路线 B 用得上。静态引入会让**所有**用户（包括只想看比例分析的）
 * 都在首屏下载它，白白多 109 kB。改成用户点了「下载并分析」时再拉。
 *
 * 缓存在模块作用域上，重复调用只会 import 一次。
 */
type ClassifierModule = typeof import('./model/classifier.ts')
let classifierModule: Promise<ClassifierModule> | null = null

function loadClassifierModule(): Promise<ClassifierModule> {
  classifierModule ??= import('./model/classifier.ts')
  return classifierModule
}

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
  model: HTMLElement
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
    model: need('model'),
  }

  /** 当前展示的图片，切换照片时释放，避免 ImageBitmap 泄漏 */
  let current: PreparedImage | null = null
  let busy = false

  /**
   * 路线 B 专用的对齐裁剪图。
   *
   * 单独留一份而不是复用 `current.bitmap`：模型是异步跑的（可能要下 54 MB），
   * 期间用户完全可能换一张照片，那时 `current` 已经被 close 了。
   * 这份跟着照片走：换照片就 close 旧的、按新的重算。
   */
  let aligned: ImageBitmap | null = null

  /**
   * 失效令牌。每次开始新的模型流程就自增；异步回调里对不上就说明
   * 用户已经换了照片，直接丢弃结果，不要把上一张的分数画到这一张上。
   */
  let modelToken = 0

  /** 本次会话已经建好模型会话 */
  let modelReady = false
  /** 缓存里已经有模型文件（挂载时查一次）。有的话就不用再问用户要不要下 */
  let modelCached = false

  function reset(): void {
    hide(el.result)
    hide(el.blocked)
    el.panel.replaceChildren()
    el.model.replaceChildren()
    current?.bitmap.close()
    current = null
    aligned?.close()
    aligned = null
    // 让还在跑的模型流程作废
    modelToken++
  }

  function showModelCard(state: ModelCardState): void {
    renderModelCard(el.model, state, { onStart: () => void runModel() }, modelReady || modelCached)
  }

  /** 跑一次路线 B。用户点按钮、或模型已就绪时自动调用。 */
  async function runModel(): Promise<void> {
    const face = aligned
    if (!face) {
      showModelCard({ kind: 'error', message: '这张照片的虹膜关键点不可用，没法做对齐裁剪。' })
      return
    }

    const token = ++modelToken
    const alive = (): boolean => token === modelToken

    try {
      // 到这一步才会去拉 onnxruntime（403 kB），之前一分钱都不花
      showModelCard({ kind: 'loading', received: 0, total: null })
      const cl = await loadClassifierModule()

      if (!cl.isLoaded()) {
        // 进度回调会来几百次（54 MB），每次重画 DOM 太浪费，
        // 只在百分比变化时才重画
        let lastPct = -1
        await cl.loadClassifier({
          onProgress: ({ received, total }) => {
            if (!alive()) return
            const pct = total && total > 0 ? Math.floor((received / total) * 100) : -2
            if (pct === lastPct) return
            lastPct = pct
            showModelCard({ kind: 'loading', received, total })
          },
        })
        modelReady = true
        modelCached = true
      }

      if (!alive()) return
      showModelCard({ kind: 'running' })

      const result = await cl.classify(face)
      if (!alive()) return
      showModelCard({ kind: 'done', result })
    } catch (err) {
      if (!alive()) return
      console.error('[good-looking] 路线 B 失败：', err)
      showModelCard({ kind: 'error', message: describeError(err) })
    }
  }

  // 挂载时查一次缓存：模型已在本地就自动跑，不必每次都问用户
  void listCachedModels()
    .then((urls) => {
      modelCached = urls.length > 0
    })
    .catch(() => {
      /* 查不到缓存不影响功能，当作没有 */
    })

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

      // 路线 B 要的是对齐裁剪后的脸。这一步很便宜（一次画布变换），
      // 先算好放着，模型卡片按需取用 —— 免得用户点「下载并分析」时
      // 还要再等一次对齐。
      const alignedFace = await alignFace(prepared.bitmap, quality.face)
      aligned = alignedFace?.bitmap ?? null

      if (!aligned) {
        showModelCard({ kind: 'error', message: '这张照片的虹膜关键点不可用，没法做对齐裁剪。' })
      } else if (modelReady || modelCached) {
        // 模型已经在本机了，就不用再问一次
        void runModel()
      } else {
        showModelCard({ kind: 'idle' })
      }
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
