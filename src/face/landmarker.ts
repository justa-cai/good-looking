/**
 * MediaPipe Face Landmarker 封装。
 *
 * 只做三件事：加载模型、跑推理、把结果整理成引擎能用的坐标系。
 * 不含任何打分逻辑 —— 那是 engine/ 的事。
 */

import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision'
import type { FaceLandmarks, Point3 } from './types.ts'

/** wasm 由 CDN 提供；版本号从 package.json 注入，与打包的 JS 严格一致。 */
const WASM_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${__MEDIAPIPE_VERSION__}/wasm`

/** Google 官方托管的 face_landmarker 模型（float16，约 3.7MB）。 */
const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'

/**
 * 最多检测几张脸。
 * 取 2 而不是 1：多人的合影需要能被识别出来并提示用户裁剪，
 * 静默挑一张打分会让用户以为结果对整张照片负责。
 */
const MAX_FACES = 2

/** MediaPipe 的后端。注意它和 ONNX Runtime 的 WebGPU/WASM 不是一回事。 */
export type Delegate = 'GPU' | 'CPU'

export type LoadStage = 'wasm' | 'model' | 'benchmark' | 'ready'

export interface LandmarkerLoadOptions {
  /**
   * 后端选择。默认 'auto'：两个后端各跑一次测速，快的留下。
   *
   * 为什么不写死一个默认值 —— 因为没法预判：
   * GPU delegate 要把结果从显存读回 CPU（ReadPixels），在软件渲染
   * （SwiftShader，比如无 GPU 的容器/虚拟机）下比 CPU 慢好几倍；
   * 但在有真实硬件加速的设备上又会快不少。这个差异只能实测。
   * 开发机上实测不出有效结论，所以让它在用户设备上自己量。
   */
  delegate?: Delegate | 'auto'
  onStage?: (stage: LoadStage) => void
  /** 后端选定后的回调，附带测得的单帧耗时（ms）。仅 'auto' 模式有耗时。 */
  onDelegateChosen?: (delegate: Delegate, msPerFrame: number) => void
  /** 'auto' 模式用来测速的图。不给则退回 CPU。 */
  probe?: ImageBitmap
}

let instance: FaceLandmarker | null = null
let pending: Promise<FaceLandmarker> | null = null
let activeDelegate: Delegate | null = null
/** 测速得到的单帧耗时，仅 'auto' 模式下有值 */
let activeMsPerFrame: number | null = null

export function getActiveDelegate(): Delegate | null {
  return activeDelegate
}

export function getActiveMsPerFrame(): number | null {
  return activeMsPerFrame
}

async function create(delegate: Delegate): Promise<FaceLandmarker> {
  const fileset = await FilesetResolver.forVisionTasks(WASM_BASE)
  return FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: MODEL_URL,
      delegate,
    },
    runningMode: 'IMAGE',
    numFaces: MAX_FACES,
    // 阈值保持默认 0.5。调高会更严格，但侧脸照会被静默丢弃 ——
    // 这属于输入质量问题，应该由 quality 层给出可解释的提示。
    outputFaceBlendshapes: false,
    outputFacialTransformationMatrixes: false,
  })
}

/** 跑一次计时的推理。首次调用含 shader 编译/内存分配，所以必须先热身。 */
function timeDetect(lm: FaceLandmarker, probe: ImageBitmap, runs = 3): number {
  lm.detect(probe) // 热身，不计时
  const t0 = performance.now()
  for (let i = 0; i < runs; i++) lm.detect(probe)
  return (performance.now() - t0) / runs
}

interface Candidate {
  delegate: Delegate
  landmarker: FaceLandmarker
  msPerFrame: number
}

/**
 * 两个后端各建一次并测速，返回更快的那个。
 * 任何一侧建不起来（GPU 在部分驱动上会失败）都不算错误，只是不参与比较。
 */
async function chooseByBenchmark(probe: ImageBitmap): Promise<Candidate> {
  const candidates: Array<Delegate> = ['GPU', 'CPU']
  let best: Candidate | null = null

  for (const delegate of candidates) {
    let landmarker: FaceLandmarker
    try {
      landmarker = await create(delegate)
    } catch (err) {
      console.warn(`[face] ${delegate} 后端初始化失败，跳过：`, err)
      continue
    }

    let msPerFrame: number
    try {
      msPerFrame = timeDetect(landmarker, probe)
    } catch (err) {
      console.warn(`[face] ${delegate} 后端推理失败，跳过：`, err)
      landmarker.close()
      continue
    }

    if (!best || msPerFrame < best.msPerFrame) {
      best?.landmarker.close()
      best = { delegate, landmarker, msPerFrame }
    } else {
      landmarker.close()
    }
  }

  if (!best) throw new Error('GPU 与 CPU 两个后端都初始化失败')
  return best
}

/**
 * 加载并缓存 FaceLandmarker。并发调用共享同一个 Promise，避免重复初始化。
 * 失败后不会把 rejected 的 Promise 留在缓存里，否则重试会一直拿到同一个错误。
 */
export async function loadLandmarker(opts: LandmarkerLoadOptions = {}): Promise<FaceLandmarker> {
  if (instance) return instance
  if (pending) return pending

  const { onStage, delegate = 'auto', probe, onDelegateChosen } = opts

  pending = (async () => {
    onStage?.('wasm')

    let chosen: Candidate

    if (delegate === 'auto' && probe) {
      onStage?.('benchmark')
      chosen = await chooseByBenchmark(probe)
      activeMsPerFrame = chosen.msPerFrame
    } else {
      // 显式指定，或 auto 但没给测速图（没图就只能保守选 CPU）
      const target: Delegate = delegate === 'auto' ? 'CPU' : delegate
      onStage?.('model')
      chosen = { delegate: target, landmarker: await create(target), msPerFrame: NaN }
    }

    activeDelegate = chosen.delegate
    onDelegateChosen?.(chosen.delegate, chosen.msPerFrame)
    onStage?.('ready')
    instance = chosen.landmarker
    return chosen.landmarker
  })()

  try {
    return await pending
  } catch (err) {
    pending = null
    throw err
  }
}

export function isLoaded(): boolean {
  return instance !== null
}

export function dispose(): void {
  instance?.close()
  instance = null
  pending = null
  activeDelegate = null
  activeMsPerFrame = null
}

function toPoint3(lm: { x: number; y: number; z: number }, width: number, height: number): Point3 {
  // z 与 x 同量纲（都按图像宽度归一化），所以也乘 width
  return { x: lm.x * width, y: lm.y * height, z: lm.z * width }
}

/**
 * 对一张已经准备好的图片跑关键点检测。
 *
 * 返回空数组表示没有检测到人脸 —— 这是正常结果，不是错误。
 * 具体怎么向用户交代由调用方决定。
 */
export function detectFaces(image: ImageBitmap): FaceLandmarks[] {
  if (!instance) throw new Error('FaceLandmarker 尚未加载，请先调用 loadLandmarker()')

  const width = image.width
  const height = image.height
  const result = instance.detect(image)

  return result.faceLandmarks.map((face) => ({
    normalized: face.map((lm) => ({ x: lm.x, y: lm.y, z: lm.z })),
    pixels: face.map((lm) => toPoint3(lm, width, height)),
    width,
    height,
  }))
}
