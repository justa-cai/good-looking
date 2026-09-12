/**
 * 应用装配层：把界面事件接到 face / engine 上，并把结果画出来。
 *
 * 这里不写任何测量或推理逻辑 —— 那些在 face/ 与 engine/ 里，且都有各自
 * 独立的验证路径。这一层只负责串起来和把话说清楚。
 *
 * 两条输入路径（上传照片 / 摄像头实时）共用同一个 `runPipeline`：冻结帧走的是
 * 和上传照片**一模一样**的流程，包括模糊检测、分项报告、对齐和学习模型。
 * 实时循环里跑的那份是精简版（跳过模糊、不跑模型），只负责取景 ——
 * 两者的差别必须在界面上说清楚，否则用户会把实时的分数当成结论。
 */

import { CameraError, startCamera, type CameraSession } from './camera/stream.ts'
import { analyze, InsufficientLandmarksError, type AnalysisResult } from './engine/index.ts'
import { alignFace } from './face/align.ts'
import {
  detectFaces,
  disposeVideo,
  getActiveDelegate,
  loadLandmarker,
  loadVideoLandmarker,
  type LoadStage,
} from './face/landmarker.ts'
import { checkQuality, type QualityReport } from './face/quality.ts'
import { prepareImage, prepareVideoFrame, type PreparedImage } from './image.ts'
import { createFrameLoop, frameDriver, type FrameLoop } from './live/loop.ts'
import { createLiveSession, type LiveFrame } from './live/session.ts'
import { listCachedModels } from './model/cache.ts'
import { createLiveView, type LiveUiState } from './ui/live.ts'
import { fmtMB, renderModelCard, type ModelCardState } from './ui/model_card.ts'
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

/**
 * 内置示例照片。
 *
 * 用途只有一个：让第一次来的访客不必先找一张自己的正脸照，点一下就能看到流程和
 * 结果长什么样。它们**就是普通输入** —— 走的是和用户自己拖进来那张一模一样的
 * `run()`，没有免检、没有专用通道。示例要是抄近路，展示的就不是真实结果了。
 *
 * 全部取自 Wikimedia Commons；`public/samples/` 里放的是裁剪缩放后的版本，
 * 原始大图不入库。作者与许可**逐张**记在 `THIRD_PARTY_NOTICES.md`，页面上还有
 * 一处可折叠的署名区 —— 两边都要补，加图时别只改一边。
 *
 * ⚠️ 这几个字段对应的是许可证义务，不是装饰：`license`/`licenseUrl`/`author`/`source`
 * 一个都不能空。署不了名的图，就不要往公开仓库里放。
 */
interface Sample {
  /** `public/samples/` 下的主名（不含扩展名）：`<base>.jpg` 全尺寸，`<base>.thumb.jpg` 缩略图 */
  readonly base: string
  readonly name: string
  readonly author: string
  readonly license: string
  readonly licenseUrl: string
  /** Commons 上的原始文件页，署名要求里的「来源」指的就是它 */
  readonly source: string
}

const SAMPLES: readonly Sample[] = [
  {
    base: '01-liu-yifei',
    name: '刘亦菲',
    author: '刘亦菲吧官方',
    license: 'CC BY-SA 2.5',
    licenseUrl: 'https://creativecommons.org/licenses/by-sa/2.5/',
    source: 'https://commons.wikimedia.org/wiki/File:Liu_Yifei_Portrait2.jpg',
  },
  {
    base: '02-yang-mi',
    name: '杨幂',
    author: 'Mercury水星記·杨幂',
    license: 'CC BY 2.5',
    licenseUrl: 'https://creativecommons.org/licenses/by/2.5/',
    source:
      'https://commons.wikimedia.org/wiki/File:%E6%9D%A8%E5%B9%82_ELLE30%E5%91%A8%E5%B9%B4%E9%A3%8E%E5%B0%9A%E5%A4%A7%E5%85%B8_%EF%BC%884%EF%BC%89.jpg',
  },
  {
    base: '03-emma-watson',
    name: 'Emma Watson',
    author: 'David Shankbone',
    license: 'CC BY 2.0',
    licenseUrl: 'https://creativecommons.org/licenses/by/2.0/',
    source: 'https://commons.wikimedia.org/wiki/File:Emma_Watson,_2012.jpg',
  },
  {
    base: '04-henry-cavill',
    name: 'Henry Cavill',
    author: 'ryanmorrisonjsy',
    license: 'CC0 1.0',
    licenseUrl: 'https://creativecommons.org/publicdomain/zero/1.0/',
    source: 'https://commons.wikimedia.org/wiki/File:Henry_Cavill-2665842_(cropped).jpg',
  },
  {
    base: '05-chris-hemsworth',
    name: 'Chris Hemsworth',
    author: 'Gage Skidmore',
    license: 'CC BY-SA 3.0',
    licenseUrl: 'https://creativecommons.org/licenses/by-sa/3.0/',
    source:
      'https://commons.wikimedia.org/wiki/File:Chris_Hemsworth_by_Gage_Skidmore_2_(cropped).jpg',
  },
  {
    base: '06-li-xian',
    name: '李现',
    author: '李现_秃头姐妹站',
    license: 'CC BY-SA 3.0',
    licenseUrl: 'https://creativecommons.org/licenses/by-sa/3.0/',
    source: 'https://commons.wikimedia.org/wiki/File:Li_Xian_3_(cropped).jpg',
  },
]

/**
 * 拼 `public/samples/` 下的地址。
 *
 * ⚠️ 必须走 `import.meta.env.BASE_URL`。站点部署在 GitHub Pages 的 `/<repo-name>/`
 * 子路径下，写死 `/samples/xxx.jpg` 在本地 dev 下看着是好的，部署后全 404 ——
 * 这个坑在 base path 那一步已经踩过一次（CI 里专门有一道校验挡它），
 * 所以这里不留例外。Vite 保证 `BASE_URL` 以 `/` 结尾。
 */
function sampleUrl(name: string): string {
  return `${import.meta.env.BASE_URL}samples/${name}`
}

/** 管线往哪儿报进度。上传路径写 #status，实时路径写实时面板自己的状态行 */
interface PipelineSink {
  status(text: string, kind?: 'busy' | 'error'): void
  hideStatus(): void
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
  picker: HTMLElement
  samples: HTMLElement
  live: HTMLElement
  tabUpload: HTMLButtonElement
  tabLive: HTMLButtonElement
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
  if (err instanceof CameraError) return err.message
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

type Mode = 'upload' | 'live'

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
    picker: need('picker'),
    samples: need('samples'),
    live: need('live'),
    tabUpload: need<HTMLButtonElement>('tab-upload'),
    tabLive: need<HTMLButtonElement>('tab-live'),
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

  /**
   * 在飞的模型流程数（下载 54 MB、建会话、推理都算）。
   *
   * 只用来判断「那张对齐图还有没有人在读」—— 见 replaceAligned。
   * 用计数而不是布尔量：`runModel` 是 fire-and-forget 的，理论上可以有两份叠着。
   */
  let modelFlows = 0

  /**
   * `cl.classify()` 正在**主线程**上同步跑。
   *
   * 和 `modelFlows` 是两件事，必须分开：只有 classify 这一步会堵住主线程
   * （`numThreads = 1`，实测 1.87s），下载和建会话都不堵。界面的锁只该覆盖
   * 前者，理由见 setModelBlocking。
   */
  let modelBlocking = false

  function reset(): void {
    hide(el.result)
    hide(el.blocked)
    el.panel.replaceChildren()
    el.model.replaceChildren()
    current?.bitmap.close()
    current = null
    replaceAligned(null)
  }

  /**
   * 换掉对齐裁剪图。
   *
   * ⚠️ 模型流程还在飞的时候**不能**立刻 close 旧的：classify 还在读它，
   * close 之后 drawImage 会抛 InvalidStateError，而 describeError 会把它
   * 误报成「这个文件没法作为图片读取」—— 一个和真实原因毫不相干的提示。
   * 那种情况下交给对应那次 runModel 的 finally 去释放（它捕获了自己的那张）。
   *
   * 交接为什么是安全的：任何一次 replaceAligned 都会让在飞的那次 `alive()` 归 false，
   * 于是它结束时一定会走到 `face.close()`；反过来，`alive()` 为 true 就说明
   * 从它开始到现在没人换过图，也就没有东西需要释放。
   */
  function replaceAligned(next: ImageBitmap | null): void {
    const old = aligned
    aligned = next
    modelToken++
    if (old && modelFlows === 0) old.close()
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

    modelFlows++
    try {
      // 到这一步才会去拉 onnxruntime（403 kB），之前一分钱都不花
      showModelCard({ kind: 'loading', received: 0, total: null })
      const cl = await loadClassifierModule()

      if (!cl.isLoaded()) {
        // 进度回调会来几百次（54 MB），每次重画 DOM 太浪费。
        // 判据是「界面上那个数字会不会变」—— 所以有总长时按百分比、没有总长时按
        // MB 档位。⚠️ 不能只按百分比：响应被压缩时（GitHub Pages 会给 .onnx 加
        // gzip，见 cache.ts）total 恒为 null，百分比就恒等于同一个值，
        // 卡片会永远停在第一次的那句话上，用户全程看不到任何进度。
        let lastKey = ''
        await cl.loadClassifier({
          onProgress: ({ received, total }) => {
            if (!alive()) return
            const key =
              total && total > 0
                ? `p${Math.floor((received / total) * 100)}`
                : `m${fmtMB(received)}`
            if (key === lastKey) return
            lastKey = key
            showModelCard({ kind: 'loading', received, total })
          },
        })
        // 即使这次流程已经被取消也要记下来：字节已经进了 Cache API，
        // 下一次冻结直接命中，这次不算白下
        modelReady = true
        modelCached = true
      }

      // ⚠️ 用户可能在下载途中点了「返回实时」—— 那时按钮是**故意**留着的，
      // 见 setModelBlocking。这里必须停住：再往下就是主线程同步的 classify，
      // 那时候实时循环会毫无征兆地卡两秒，正是这个设计要避免的事。
      if (!alive()) return
      showModelCard({ kind: 'running' })

      setModelBlocking(true)
      try {
        // 让浏览器先把上一步的 DOM 改动画出去。classify 一进去就是同步阻塞，
        // 不给一次任务的间隙的话，「推理中」和禁用的按钮根本没有机会上屏 ——
        // 用户看到的是界面直接卡住，而不是「正在推理」
        await new Promise((resolve) => setTimeout(resolve, 0))

        const result = await cl.classify(face)
        if (!alive()) return
        showModelCard({ kind: 'done', result })
      } finally {
        setModelBlocking(false)
      }
    } catch (err) {
      if (!alive()) return
      console.error('[good-looking] 路线 B 失败：', err)
      showModelCard({ kind: 'error', message: describeError(err) })
    } finally {
      modelFlows--
      setModelBlocking(false)
      // 这张对齐图已经不是当前的了（用户换了照片），由这里负责释放。
      // 见 replaceAligned 的说明
      if (!alive()) face.close()
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

  // ==================== 渲染管线 ====================
  //
  // 从「已经拿到 PreparedImage」开始的一切：检测 → 质量校验 → 打分 → 画图 →
  // 对齐 → 路线 B。上传照片和实时冻结帧走的是同一份，这是刻意的 ——
  // 冻结的全部意义就是「把此刻看到的那一帧按照片的方式仔细算一遍」，
  // 两条路径一旦分叉，同一个取景会给出两个结论。

  async function runPipeline(prepared: PreparedImage, sink: PipelineSink): Promise<void> {
    sink.status('正在定位人脸…')
    const faces = detectFaces(prepared.bitmap)

    sink.status('正在校验照片质量…')
    // 传位图是为了做模糊检测 —— 那一项要读原始像素，关键点里没有
    const quality = checkQuality(faces, prepared.bitmap)

    if (!quality.acceptable || !quality.face) {
      current = prepared
      renderQuality(el.blocked, quality)
      sink.hideStatus()
      return
    }

    // 还有警告也一并展示，但继续出分
    renderQuality(el.blocked, quality)

    let result: AnalysisResult
    try {
      result = analyze(quality.face.pixels)
    } catch (err) {
      current = prepared
      sink.status(describeError(err), 'error')
      return
    }

    current = prepared
    drawOverlay(el.canvas, prepared.bitmap, quality.face, result)
    renderReport(el.panel, result)

    el.result.hidden = false
    sink.hideStatus()

    // 路线 B 要的是对齐裁剪后的脸。这一步很便宜（一次画布变换），
    // 先算好放着，模型卡片按需取用 —— 免得用户点「下载并分析」时
    // 还要再等一次对齐。
    const alignedFace = await alignFace(prepared.bitmap, quality.face)
    replaceAligned(alignedFace?.bitmap ?? null)

    if (!aligned) {
      showModelCard({ kind: 'error', message: '这张照片的虹膜关键点不可用，没法做对齐裁剪。' })
    } else if (modelReady || modelCached) {
      // 模型已经在本机了，就不用再问一次
      void runModel()
    } else {
      showModelCard({ kind: 'idle' })
    }
  }

  // ==================== 模式切换 ====================

  let mode: Mode = 'upload'

  function switchMode(next: Mode): void {
    if (next === mode) return
    mode = next

    el.picker.hidden = next !== 'upload'
    el.live.hidden = next !== 'live'
    el.tabUpload.classList.toggle('is-active', next === 'upload')
    el.tabLive.classList.toggle('is-active', next === 'live')
    el.tabUpload.setAttribute('aria-selected', String(next === 'upload'))
    el.tabLive.setAttribute('aria-selected', String(next === 'live'))

    if (next === 'upload') {
      // 离开实时模式就把摄像头关掉 —— 指示灯不能因为切了个标签就一直亮着
      stopLive()
      hide(el.status)
    } else {
      // 进实时模式先清掉上一次的结果：#result 和实时读数同屏就是两个分数
      reset()
    }
  }

  el.tabUpload.addEventListener('click', () => switchMode('upload'))
  el.tabLive.addEventListener('click', () => switchMode('live'))

  // ==================== 摄像头实时 ====================

  const live = createLiveView(el.live, {
    onStart: () => void startLive(),
    onStop: () => stopLive(),
    onFreeze: () => void freezeFrame(),
    onBackToLive: () => backToLive(),
  })
  const liveSession = createLiveSession(live.video)

  let loop: FrameLoop | null = null
  let camera: CameraSession | null = null
  let liveState: LiveUiState = 'idle'
  /**
   * 冻结结果是否已经上屏。
   *
   * 和 `liveState === 'working'` 不是一回事：那个状态在冻结**过程中**也会用到
   * （管线还在跑）。这里要区分的是「结果已经摆在用户面前了」，因为只有到那一步
   * 才有「返回实时」可点，才谈得上被模型堵住。
   */
  let freezeResultShown = false

  const MODEL_BUSY_TEXT = '学习模型正在推理，实时测量已暂停 —— 大约两秒'

  /**
   * `cl.classify()` 的起落 —— 也就是**主线程真正被堵住**的那一段的起落。
   *
   * 只有这一段该把实时面板锁住：它跑在 `numThreads = 1` 的 WASM 上，实测 **2.4~2.5 s**
   * （PerformanceObserver 的长任务条目，本机），期间整个页面都不响应。这时如果
   * 「返回实时」还点得动，用户点下去会毫无征兆地卡两秒多，看起来像程序挂了。
   *
   * 下载那一大段（54 MB，慢网络下几分钟）主线程是空闲的，直播循环照常跑，
   * 所以**不能**把按钮禁掉 —— 实测踩过这个坑：CDN 上的 `ort-wasm-*.wasm` 有一次
   * 拿到 200 之后响应体再也没下完，界面就那样挂着，除了刷新页面没有别的出路。
   * 那段时间「返回实时」是开着的，而且点了就真的取消得掉：`backToLive` 会走
   * `reset()` → `replaceAligned` → 令牌自增，runModel 在 classify 之前的那次
   * `alive()` 检查就过不去了。实测：点一下之后全程长任务 0 个、evaluate 往返最大 94 ms。
   *
   * ⚠️ 一个诚实的补充：**收完 54 MB 的那一刻**主线程会停一下（本机实测 1.6 s），
   * 取消也避不开。原因在 cache.ts —— `readWithProgress` 要把上百个分片拼成一个
   * 连续 ArrayBuffer，`writeCache` 又 `slice(0)` 了一份，两次 54 MB 的同步拷贝。
   * 它和 classify 是两回事（那个是我们能控制的、必须如实告知用户的两秒多），
   * 所以这里只锁 classify，那一次停顿另行记录。
   */
  function setModelBlocking(on: boolean): void {
    modelBlocking = on
    // 没在冻结态就没什么可锁的（上传那条路的进行状态由 #model 卡片自己说）
    if (liveState === 'idle' || !freezeResultShown) return
    liveState = on ? 'working' : 'frozen'
    live.setState(liveState)
    live.setStatus(on ? MODEL_BUSY_TEXT : null)
  }

  async function startLive(): Promise<void> {
    if (liveState !== 'idle') return
    liveState = 'starting'
    live.setState('starting')
    live.setError(null)
    live.setStatus('正在打开摄像头…')

    try {
      camera = await startCamera(live.video, {
        onEnded: () => {
          // 断流之后 <video> 会停在最后一帧不动：画面看着是好的，
          // 而循环还在以 10Hz 把同一组关键点喂进平滑窗口，
          // 最后给出一个又稳又死的分数。必须立刻停
          live.setError('摄像头断开了（被拔掉、或被别的程序抢走了）。请重新点「开始」。')
          stopLive()
        },
      })
    } catch (err) {
      // CameraError 已经在 stream.ts 里连同环境诊断打过一条了，这儿就不再重复
      // 一条信息量更少的 —— 同一个失败在控制台里出现两次只会让人以为是两回事
      if (!(err instanceof CameraError)) console.error('[good-looking] 打开摄像头失败：', err)
      liveState = 'idle'
      live.setState('idle')
      live.setStatus(null)
      live.setError(describeError(err))
      return
    }

    let probe: ImageBitmap | null = null
    try {
      // 第一次加载时后端还没定，得给测速一张图。摄像头帧正好能用 ——
      // 这样「先开摄像头、还没传过照片」这条路径也能选到正确的后端
      if (!getActiveDelegate()) probe = await createImageBitmap(live.video)

      live.setStatus(STAGE_TEXT.wasm)
      await loadVideoLandmarker({
        delegate: 'auto',
        ...(probe ? { probe } : {}),
        onStage: (stage) => {
          const text = STAGE_TEXT[stage]
          if (text) live.setStatus(text)
        },
      })
    } catch (err) {
      console.error('[good-looking] 加载检测引擎失败：', err)
      probe?.close()
      stopLive()
      live.setError(describeError(err))
      return
    }
    probe?.close()

    live.setDriver(frameDriver(live.video))
    liveSession.reset()
    loop = createFrameLoop(live.video, {
      onFrame: (now, deltaMs) => {
        const frame = liveSession.frame(now, deltaMs)
        drawLiveFrame(frame)
        live.update(frame)
      },
      onError: (err) => {
        // 单帧失败是正常的（画布正在 resize、某一帧关键点不足），只记日志
        console.warn('[good-looking] 实时帧失败：', err)
      },
      onFatal: (err) => handleLiveFatal(err),
      onVisibilityChange: (hidden) => {
        // 切回来的第一帧和隐藏前的样本之间可能隔了几分钟，窗口必须重来
        if (!hidden) liveSession.reset()
      },
    })

    liveState = 'running'
    live.setState('running')
    live.setStatus(null)
    loop.start()
  }

  function drawLiveFrame(frame: LiveFrame): void {
    if (!frame.face || !frame.displayPoints) return
    // result 传 null：实时模式不展示分项颜色，测量段一律灰色。
    // 这里的目标是「看清楚自己的脸在哪儿」，颜色会变成噪声
    drawOverlay(live.canvas, live.video, frame.face, null, {
      points: frame.displayPoints,
      showAllPoints: false,
    })
  }

  function handleLiveFatal(err: unknown): void {
    console.error('[good-looking] 实时检测连续失败：', err)
    // MediaPipe 的时间戳违例会让 C++ graph **永久卡死**，之后每帧都抛同样的错。
    // 重建实例是唯一的出路，重试没有意义
    disposeVideo()
    stopLive()
    live.setError(
      `实时检测连续失败，已经停下（${err instanceof Error ? err.message : String(err)}）。请重新点「开始」。`,
    )
  }

  function stopLive(): void {
    loop?.stop()
    loop = null
    camera?.stop()
    camera = null
    liveSession.reset()
    freezeResultShown = false
    live.update(null)
    live.setError(null)
    live.setStatus(null)
    live.setState('idle')
    liveState = 'idle'
  }

  async function freezeFrame(): Promise<void> {
    if (liveState !== 'running') return
    liveState = 'working'
    freezeResultShown = false
    live.setState('working')
    live.setStatus('正在冻结这一帧…')

    loop?.stop()
    loop = null
    live.video.pause()

    try {
      // 走和上传照片完全相同的处理：同样的降采样上限、同样的坐标语义
      const prepared = await prepareVideoFrame(live.video)
      reset()
      await runPipeline(prepared, {
        status: (text, kind) => live.setStatus(text, kind),
        hideStatus: () => live.setStatus(null),
      })
      // 结果已经上屏。模型可能正跑着，但它此刻可能还在**下载**那一段
      // （主线程空闲），那种情况下状态是 'frozen' ——「返回实时」要留着，
      // 因为下载可能几分钟，而用户随时可以取消掉这条流程。
      // 真正开始堵主线程时 setModelBlocking 会把状态翻成 'working'
      freezeResultShown = true
      liveState = modelBlocking ? 'working' : 'frozen'
      live.setState(liveState)
      live.setStatus(modelBlocking ? MODEL_BUSY_TEXT : null)
    } catch (err) {
      console.error('[good-looking] 冻结这一帧失败：', err)
      freezeResultShown = false
      live.setError(describeError(err))
      stopLive()
    }
  }

  function backToLive(): void {
    if (liveState !== 'frozen' || !camera) return
    // 冻结结果占着 #result。回实时就把它清掉 —— 结构上同一时间只能有一个分数
    freezeResultShown = false
    reset()
    liveState = 'running'
    live.setState('running')
    liveSession.reset()
    live.setError(null)
    live.setStatus(null)
    void live.video.play().catch(() => {
      /* 自动播放被拒（少见，muted 之后一般不会）只影响画面，循环照样会重试 */
    })
    loop?.start()
  }

  window.addEventListener('pagehide', () => {
    stopLive()
  })

  // ==================== 上传照片 ====================

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

      await runPipeline(prepared, {
        status: (text, kind) => showStatus(el.status, text, kind),
        hideStatus: () => hide(el.status),
      })
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

  // ==================== 示例照片 ====================

  /** 示例照片正在取字节。和 `busy`（管线在跑）分开：取回字节之前管线还没开始 */
  let samplesLoading = false

  /**
   * 取一张示例照片来跑。
   *
   * 只是「把字节取回来」这一步是示例独有的，之后就全部交给 `run()` ——
   * 和用户自己选文件走的是同一条路。
   *
   * ⚠️ 不能只看 `res.ok`。这个项目已经踩过两次同一类坑（CLAUDE.md 里记着）：
   * 静态服务器对不存在的路径返回 **200 + index.html** 而不是 404 是很常见的配置，
   * 那样拿到的是一页 HTML，最后报出来的却是
   * `InvalidStateError: The source image could not be decoded` —— 看起来像图片格式
   * 有问题，其实只是路径错了。所以状态码之外还要看 content-type 是不是图片。
   */
  async function runSample(sample: Sample): Promise<void> {
    if (busy || samplesLoading) return
    samplesLoading = true
    el.samples.classList.add('is-busy')
    try {
      const res = await fetch(sampleUrl(`${sample.base}.jpg`))
      if (!res.ok) throw new Error(`示例照片读取失败（HTTP ${res.status}）。`)
      const blob = await res.blob()
      if (!blob.type.startsWith('image/')) {
        throw new Error('示例照片读取失败：服务器返回的不是图片。')
      }
      // run() 收 File。示例照片取回来的只是一个 Blob，这里补上文件名和类型 ——
      // 除此之外它和用户拖进来的文件没有任何区别
      await run(new File([blob], `${sample.base}.jpg`, { type: blob.type }))
    } catch (err) {
      // 这里不走 describeError：上面两句本身就是要给用户看的话，
      // 再套一层「处理失败：」只会把话说糊
      console.error('[good-looking] 读取示例照片失败：', err)
      showStatus(el.status, err instanceof Error ? err.message : String(err), 'error')
    } finally {
      samplesLoading = false
      el.samples.classList.remove('is-busy')
    }
  }

  /** 署名区。CC BY / CC BY-SA 都要求署名，CC0 那张也一并列上 —— 分两处写容易漏 */
  function buildSampleCredits(): HTMLElement {
    const details = document.createElement('details')
    details.className = 'samples__credits'

    const summary = document.createElement('summary')
    summary.textContent = '示例照片的作者与许可'
    details.append(summary)

    const ul = document.createElement('ul')
    for (const s of SAMPLES) {
      const li = document.createElement('li')

      const lic = document.createElement('a')
      lic.href = s.licenseUrl
      lic.target = '_blank'
      lic.rel = 'noopener noreferrer'
      lic.textContent = s.license

      const src = document.createElement('a')
      src.href = s.source
      src.target = '_blank'
      src.rel = 'noopener noreferrer'
      src.textContent = '原始文件页'

      li.append(`${s.name}：${s.author}，`, lic, '，', src)
      ul.append(li)
    }
    details.append(ul)

    const note = document.createElement('p')
    note.className = 'samples__note'
    note.textContent =
      '这些都是真实人物的公开活动照，在这里只作为分析的输入使用，' +
      '不代表对照片中任何人的评价。入库的是裁剪缩放后的版本（改编作品），' +
      '仍按原许可再分发；去掉作者署名或改许可都不行，明细见仓库里的 THIRD_PARTY_NOTICES.md。'
    details.append(note)

    return details
  }

  function buildSamples(): void {
    const label = document.createElement('p')
    label.className = 'samples__label'
    label.textContent = '没有合适的照片？用这张示例试一下：'

    const list = document.createElement('div')
    list.className = 'samples__list'
    for (const s of SAMPLES) {
      const btn = document.createElement('button')
      btn.type = 'button'
      btn.className = 'samples__item'
      btn.title = `${s.name} · 摄影 ${s.author} · ${s.license}`

      const img = document.createElement('img')
      img.className = 'samples__thumb'
      img.src = sampleUrl(`${s.base}.thumb.jpg`)
      // 名字就在下面一行，读屏再念一遍是重复
      img.alt = ''
      img.loading = 'lazy'
      img.decoding = 'async'
      img.width = 88
      img.height = 88

      const name = document.createElement('span')
      name.className = 'samples__name'
      name.textContent = s.name

      btn.append(img, name)
      btn.addEventListener('click', () => void runSample(s))
      list.append(btn)
    }

    el.samples.replaceChildren(label, list, buildSampleCredits())
  }

  buildSamples()
}
