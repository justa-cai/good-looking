/**
 * 路线 B 的推理封装：ViT 二分类器（onnxruntime-web）。
 *
 * 只做三件事：加载模型、选后端、把裁剪好的脸跑成概率。
 * 不含任何打分/换算逻辑 —— 模型输出「怎么用」是界面层的事，
 * 而且这个模型有严重的人群偏移（见 CLAUDE.md），换算规则必须单独想清楚。
 *
 * ⚠️ 输入必须是对齐裁剪后的 224×224 图（见 `src/face/align.ts`）。
 * 直接喂整张照片的后果有实测数据：同一人三张照片的极差从 0.215 涨到 0.671。
 */

import * as ort from 'onnxruntime-web'

/**
 * ONNX Runtime 的 wasm 二进制走 CDN，不打包进产物。
 *
 * 理由和 MediaPipe 那边一样：wasm 有十几 MB，放 CDN 能让仓库和 Pages 产物
 * 都保持轻量，而且 jsDelivr 的缓存命中率比 GitHub Pages 高。
 * 版本号写死成和 package.json 里的依赖一致 —— 差了会静默用错版本。
 */
const ORT_VERSION = '1.29.0'
const ORT_WASM_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`

/** 模型固定的输入边长。导出时 H/W 就是静态的，喂别的尺寸会直接报错。 */
export const INPUT_SIZE = 224

/** 后端。对应 ORT 的 executionProviders。 */
export type Provider = 'webgpu' | 'wasm'

export interface ClassifierLoadOptions {
  /** 模型地址。托管策略见任务 #12；不给则用默认路径。 */
  url?: string
  /**
   * 后端选择。默认 'auto'：可用就两个都建、各测一次速，快的留下。
   *
   * 和 MediaPipe 那边同样的理由 —— 没法预判。WebGPU 在真实硬件上通常更快，
   * 但在软件渲染（SwiftShader）下会慢好几倍，而这个差异只能在用户设备上量。
   */
  provider?: Provider | 'auto'
  onStage?: (stage: ClassifierLoadStage) => void
  /** 后端选定后的回调，附带实测单次推理耗时（ms）。仅 'auto' 模式有耗时。 */
  onProviderChosen?: (provider: Provider, msPerRun: number) => void
  /** 'auto' 模式用来测速的脸图（对齐后的裁剪）。不给则退回 WASM。 */
  probe?: ImageBitmap
}

export type ClassifierLoadStage = 'model' | 'benchmark' | 'ready'

export interface Classification {
  /** 「attractive」这一类的概率（原始 softmax，未经任何校准） */
  readonly pAttractive: number
  /**
   * 两类 logits。保留原始值是因为这个模型的绝对概率有严重人群偏移
   * （见 CLAUDE.md「绝对概率不能当分数用」），做相对比较时用 logit 差更稳：
   * 它没有被 softmax 压扁，跨样本的区分度更好。
   */
  readonly logits: readonly [number, number]
}

let session: ort.InferenceSession | null = null
let pending: Promise<ort.InferenceSession> | null = null
let activeProvider: Provider | null = null
let activeMsPerRun: number | null = null
/** 一次只跑一个人，复用同一块画布，省掉每次分配 */
let scratch: HTMLCanvasElement | null = null

export function getActiveProvider(): Provider | null {
  return activeProvider
}

export function getActiveMsPerRun(): number | null {
  return activeMsPerRun
}

export function isLoaded(): boolean {
  return session !== null
}

/**
 * 把位图编码成模型输入张量。
 *
 * 归一化系数来自模型的 preprocessor_config.json：
 * resize 到 224（bilinear）→ 除以 255 → 减均值 0.5、除标准差 0.5。
 * 合并起来就是 `x/127.5 - 1`，写成一步省一次遍历。
 *
 * ⚠️ 这三个数字必须和 Python 端 `AutoImageProcessor` 一致，否则数值对不上。
 * 改之前先跑 scripts/model/export_onnx.py 的真图比对 —— 它用的就是
 * 官方 processor，对不上会直接暴露成 max|Δlogits| 变大。
 */
function toTensor(bitmap: ImageBitmap): ort.Tensor {
  if (!scratch) scratch = document.createElement('canvas')
  if (scratch.width !== INPUT_SIZE || scratch.height !== INPUT_SIZE) {
    scratch.width = INPUT_SIZE
    scratch.height = INPUT_SIZE
  }

  const ctx = scratch.getContext('2d', { willReadFrequently: true })
  if (!ctx) throw new Error('拿不到 2D 上下文，无法编码模型输入')

  ctx.imageSmoothingEnabled = true
  ctx.imageSmoothingQuality = 'high'
  ctx.drawImage(bitmap, 0, 0, INPUT_SIZE, INPUT_SIZE)
  const { data } = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE)

  // ONNX 要 NCHW，而且第一个是 batch 维 —— 这里固定 batch=1
  const plane = INPUT_SIZE * INPUT_SIZE
  const out = new Float32Array(3 * plane)
  for (let i = 0, p = 0; i < plane; i++, p += 4) {
    out[i] = data[p]! / 127.5 - 1
    out[plane + i] = data[p + 1]! / 127.5 - 1
    out[2 * plane + i] = data[p + 2]! / 127.5 - 1
  }

  return new ort.Tensor('float32', out, [1, 3, INPUT_SIZE, INPUT_SIZE])
}

async function timeRun(sess: ort.InferenceSession, bitmap: ImageBitmap, runs = 2): Promise<number> {
  const tensor = toTensor(bitmap)
  await sess.run({ pixel_values: tensor }) // 热身：首次含显存/内存分配
  const t0 = performance.now()
  for (let i = 0; i < runs; i++) await sess.run({ pixel_values: tensor })
  return (performance.now() - t0) / runs
}

async function create(provider: Provider, url: string): Promise<ort.InferenceSession> {
  return ort.InferenceSession.create(url, {
    executionProviders: [provider],
    // 图优化开到最大：这是离线量化的静态图，可以做常量折叠等重写
    graphOptimizationLevel: 'all',
  })
}

interface Candidate {
  provider: Provider
  session: ort.InferenceSession
  msPerRun: number
}

/**
 * 可用就两个后端各建一次并测速，返回更快的那个。
 * 任何一侧建不起来（部分环境没有 WebGPU）都不算错误，只是不参与比较。
 */
async function chooseByBenchmark(url: string, probe: ImageBitmap): Promise<Candidate> {
  // 没有 WebGPU 就别浪费时间建 —— 建失败会抛，但那是预期内的
  const hasWebGpu = typeof navigator !== 'undefined' && 'gpu' in navigator
  const wanted: Provider[] = hasWebGpu ? ['webgpu', 'wasm'] : ['wasm']

  let best: Candidate | null = null

  for (const provider of wanted) {
    let sess: ort.InferenceSession
    try {
      sess = await create(provider, url)
    } catch (err) {
      console.warn(`[model] ${provider} 后端初始化失败，跳过：`, err)
      continue
    }

    let msPerRun: number
    try {
      msPerRun = await timeRun(sess, probe)
    } catch (err) {
      console.warn(`[model] ${provider} 后端推理失败，跳过：`, err)
      await sess.release?.()
      continue
    }

    if (!best || msPerRun < best.msPerRun) {
      await best?.session.release?.()
      best = { provider, session: sess, msPerRun }
    } else {
      await sess.release?.()
    }
  }

  if (!best) throw new Error('没有可用的推理后端')
  return best
}

/**
 * 加载并缓存分类器。并发调用共享同一个 Promise，避免重复初始化。
 * 失败后不会把 rejected 的 Promise 留在缓存里，否则重试会一直拿到同一个错误。
 */
export async function loadClassifier(
  opts: ClassifierLoadOptions = {},
): Promise<ort.InferenceSession> {
  if (session) return session
  if (pending) return pending

  const { url, provider = 'auto', onStage, onProviderChosen } = opts
  const modelUrl = url ?? defaultModelUrl()

  pending = (async () => {
    ort.env.wasm.wasmPaths = ORT_WASM_BASE
    // ⚠️ 强制单线程。GitHub Pages 没有 COOP/COEP，SharedArrayBuffer 不可用，
    // 多线程会自动退化成单线程 —— 但退化路径和显式单线程的行为并不完全一样
    // （内存分配、构建产物都不同）。显式写死，保证本地和线上表现一致，
    // 避免出现「本地飞快、线上慢十倍」这种只在部署后才暴露的问题。
    ort.env.wasm.numThreads = 1
    ort.env.logLevel = 'error'

    onStage?.('model')

    let chosen: Candidate
    if (provider === 'auto' && opts.probe !== undefined) {
      onStage?.('benchmark')
      chosen = await chooseByBenchmark(modelUrl, opts.probe)
      activeMsPerRun = chosen.msPerRun
    } else if (provider === 'auto') {
      // auto 但没给测速图：没图就只能保守选 WASM（它一定可用）
      chosen = { provider: 'wasm', session: await create('wasm', modelUrl), msPerRun: NaN }
    } else {
      chosen = { provider, session: await create(provider, modelUrl), msPerRun: NaN }
    }

    activeProvider = chosen.provider
    onProviderChosen?.(chosen.provider, chosen.msPerRun)
    onStage?.('ready')
    session = chosen.session
    return chosen.session
  })()

  try {
    return await pending
  } catch (err) {
    pending = null
    throw err
  }
}

/** 默认模型地址。托管策略定下来之前先指到同源的 public/models/。 */
function defaultModelUrl(): string {
  // import.meta.env.BASE_URL 会带上 /good-looking/ 前缀，
  // 硬编码 '/' 在 GitHub Pages 上会 404
  const base = import.meta.env.BASE_URL
  return `${base}models/attractive.int4.onnx`
}

/**
 * 对一张脸出概率。
 *
 * @param face 对齐裁剪后的位图（`alignFace` 的输出），必须已是 224×224 量级；
 *   尺寸不对会被拉伸，不会报错，但结果不可信。
 */
export async function classify(face: ImageBitmap): Promise<Classification> {
  if (!session) throw new Error('分类器尚未加载，请先调用 loadClassifier()')

  const out = await session.run({ pixel_values: toTensor(face) })
  const raw = out.logits?.data
  // Tensor.data 是若干种 TypedArray 的联合类型，先收紧再算 ——
  // 顺带把「模型给错了 dtype」这种问题变成明确的报错，而不是 NaN。
  if (!(raw instanceof Float32Array) || raw.length < 2) {
    throw new Error(`模型输出形状不对，期望 2 个 float32 logits，实际 ${raw?.length ?? 0}`)
  }

  const a = raw[0]!
  const b = raw[1]!
  const m = Math.max(a, b)
  const ea = Math.exp(a - m)
  const eb = Math.exp(b - m)

  return { pAttractive: ea / (ea + eb), logits: [a, b] }
}

export async function dispose(): Promise<void> {
  await session?.release?.()
  session = null
  pending = null
  activeProvider = null
  activeMsPerRun = null
  scratch = null
}
