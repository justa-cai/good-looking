/**
 * MediaPipe Face Landmarker 封装。
 *
 * 只做三件事：加载模型、跑推理、把结果整理成引擎能用的坐标系。
 * 不含任何打分逻辑 —— 那是 engine/ 的事。
 */

import { FaceLandmarker, FilesetResolver, type FaceLandmarkerResult } from '@mediapipe/tasks-vision'
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

/**
 * 视频路径只取一张脸。
 *
 * 和静态照片的取舍相反，是因为场景不同：quality.ts 对「多于一张脸」是整帧拒判
 * （对照片是对的 —— 用户无法知道分数对应的是哪个人），但在视频里，墙上海报、
 * 玻璃反光、路过的人都会让整帧拒判反复触发，不断清空平滑窗口，分数永远稳不下来，
 * 而界面会暗示是用户自己的问题。所以实时路径只跟踪最主要的那张脸，并由界面明说。
 */
const VIDEO_MAX_FACES = 1

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

/**
 * 视频帧检测用的**独立**实例。
 *
 * 为什么不复用上面那个单例的 `setOptions({runningMode})` 切换模式：
 *
 * 1. `detect()` 在 VIDEO 模式下会抛错、`detectForVideo()` 在 IMAGE 模式下会抛错
 *    （两个方向都有硬性守卫）。而上传 tab 的 file input 与拖放监听**始终活着**，
 *    所以只要实时循环把单例翻成 VIDEO，冻结帧和之后上传的任何照片都会直接抛错。
 * 2. `setOptions()` 是异步且没有串行化的，循环恢复和冻结并发时会互相踩。
 * 3. `loadLandmarker` 的 'auto' 测速内部要用 IMAGE 模式的 `detect()`，提前翻模式
 *    会让两个后端都测速失败，报出误导性的「两个后端都初始化失败」。
 *
 * 多建一个实例与既有实践一致 —— `chooseByBenchmark` 本来就创建/销毁多个实例。
 * 代价是第二个图与 tflite 解释器常驻，模型文件本身走 HTTP 缓存不会重下。
 */
let videoInstance: FaceLandmarker | null = null
let videoPending: Promise<FaceLandmarker> | null = null

/**
 * 视频路径的时间戳高水位。
 *
 * MediaPipe 的 C++ graph 强制时间戳**严格递增**，违反是硬错误：graph 会卡死，
 * 之后每次 `detectForVideo` 都抛，直到重建实例。两种最容易踩的触发方式是
 * 混用时钟（rVFC 的 `now` 是 performance 时间线，`metadata.mediaTime` 是从 0 起的
 * 媒体时间线）和流重启后重置计数（新流的 mediaTime 回到 0，而 graph 还记着旧的
 * 最高水位）。所以这里只做两件事：用一个全局单调的计数器，且**永不重置** ——
 * 换流、重开摄像头、甚至重建实例之后都继续往前走，往前走不会有副作用。
 */
let videoTimestamp = Number.NEGATIVE_INFINITY

export function getActiveDelegate(): Delegate | null {
  return activeDelegate
}

export function getActiveMsPerFrame(): number | null {
  return activeMsPerFrame
}

async function create(delegate: Delegate, runningMode: 'IMAGE' | 'VIDEO'): Promise<FaceLandmarker> {
  const fileset = await FilesetResolver.forVisionTasks(WASM_BASE)
  return FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: MODEL_URL,
      delegate,
    },
    runningMode,
    numFaces: runningMode === 'VIDEO' ? VIDEO_MAX_FACES : MAX_FACES,
    // 阈值保持默认 0.5。调高会更严格，但侧脸照会被静默丢弃 ——
    // 这属于输入质量问题，应该由 quality 层给出可解释的提示。
    outputFaceBlendshapes: false,
    // 打开：头部姿态只能从这里拿。关键点本身推不出可靠的姿态
    // （试过用左右关键点的对称性当判据，噪声太大，见 CLAUDE.md）。
    outputFacialTransformationMatrixes: true,
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
 * 两个后端全失败时的说法。
 *
 * ⚠️ 别直接抛「两个后端都初始化失败」：实测最常见的真实原因是**资源根本没下下来**
 * （离线、CDN 被挡、代理断流），而 MediaPipe 的 wasm/tflite 加载失败抛的是
 * `Event` 而不是 `Error` —— 消息是空的，冒到界面上就成了一句
 * 「GPU 与 CPU 两个后端都初始化失败」，用户完全不知道该去查网络。
 * 这和 CLAUDE.md 里记的那两个坑是同一类：报错信息指向的地方不是真正出问题的地方。
 *
 * 判据「抛出来的不是 Error」是个**启发式**，不是铁证：真遇到 wasm 构建本身坏掉
 * 也会是这个样子。所以措辞用「最常见的原因是…」，不把话说死。
 */
function describeTotalFailure(errors: readonly unknown[]): string {
  const opaque = errors.length > 0 && errors.every((e) => !(e instanceof Error))
  if (opaque) {
    return '人脸检测引擎没能加载：GPU 与 CPU 两个后端都失败了，最常见的原因是模型文件没下载成功。请检查网络（或代理）后重试。'
  }
  return 'GPU 与 CPU 两个后端都初始化失败'
}

/**
 * 两个后端各建一次并测速，返回更快的那个。
 * 任何一侧建不起来（GPU 在部分驱动上会失败）都不算错误，只是不参与比较。
 */
async function chooseByBenchmark(probe: ImageBitmap): Promise<Candidate> {
  const candidates: Array<Delegate> = ['GPU', 'CPU']
  let best: Candidate | null = null
  const failures: unknown[] = []

  for (const delegate of candidates) {
    let landmarker: FaceLandmarker
    try {
      landmarker = await create(delegate, 'IMAGE')
    } catch (err) {
      console.warn(`[face] ${delegate} 后端初始化失败，跳过：`, err)
      failures.push(err)
      continue
    }

    let msPerFrame: number
    try {
      msPerFrame = timeDetect(landmarker, probe)
    } catch (err) {
      console.warn(`[face] ${delegate} 后端推理失败，跳过：`, err)
      failures.push(err)
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

  if (!best) throw new Error(describeTotalFailure(failures))
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
      chosen = { delegate: target, landmarker: await create(target, 'IMAGE'), msPerFrame: NaN }
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
  // 视频实例也要一起清掉：它是按 activeDelegate 建的，把后端决定清空却留着它，
  // 会让下一次 loadVideoLandmarker 拿到一个后端来历不明的旧实例。
  disposeVideo()
}

function toPoint3(lm: { x: number; y: number; z: number }, width: number, height: number): Point3 {
  // z 与 x 同量纲（都按图像宽度归一化），所以也乘 width
  return { x: lm.x * width, y: lm.y * height, z: lm.z * width }
}

/**
 * MediaPipe 原始结果 → 引擎能用的坐标系。两条路径（静态图 / 视频帧）共用这一份。
 *
 * 抽出来是因为它属于 CLAUDE.md 里点名要当心的地方：关键点索引语义一旦两条路径
 * 不一致，错误不会报出来，只会让所有人偏丑或偏好看。一份实现就没有分叉的可能。
 */
function toFaceLandmarks(
  result: FaceLandmarkerResult,
  width: number,
  height: number,
): FaceLandmarks[] {
  return result.faceLandmarks.map((face, i) => {
    const matrix = result.facialTransformationMatrixes[i]
    return {
      normalized: face.map((lm) => ({ x: lm.x, y: lm.y, z: lm.z })),
      pixels: face.map((lm) => toPoint3(lm, width, height)),
      width,
      height,
      transform: matrix ? Array.from(matrix.data) : null,
    }
  })
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
  return toFaceLandmarks(instance.detect(image), width, height)
}

/**
 * 建立（或复用）视频帧检测用的实例。
 *
 * 会先把静态图那个实例加载好 —— 后端选择是**一次性的全局决定**，由
 * `loadLandmarker` 的测速做出，视频实例只是沿用，不重新测速。这样也保证了
 * 「先开摄像头、还没上传过照片」这条路径有正确的后端（此时需要调用方把视频帧
 * 当 probe 传进来，否则按既有约定退回 CPU）。
 */
export async function loadVideoLandmarker(
  opts: LandmarkerLoadOptions = {},
): Promise<FaceLandmarker> {
  if (videoInstance) return videoInstance
  if (videoPending) return videoPending

  videoPending = (async () => {
    // 顺序很重要：这个调用内部会用 IMAGE 模式的 detect() 测速。
    await loadLandmarker(opts)
    if (!activeDelegate) throw new Error('后端尚未确定，无法建立视频检测实例')

    const created = await create(activeDelegate, 'VIDEO')
    videoInstance = created
    return created
  })()

  try {
    return await videoPending
  } catch (err) {
    videoPending = null
    throw err
  }
}

export function isVideoLoaded(): boolean {
  return videoInstance !== null
}

/**
 * 关掉视频实例。循环连续失败时调用它，下次 `loadVideoLandmarker` 会重建一个
 * 干净的图 —— 时间戳违例会让 graph 永久卡死，重建是唯一的出路。
 * 注意**不清空** `videoTimestamp`：新图的计数器从 0 开始也没关系，
 * 继续往前走同样没关系，而保留它能少一类边界情况。
 */
export function disposeVideo(): void {
  videoInstance?.close()
  videoInstance = null
  videoPending = null
}

/**
 * 对一帧视频跑关键点检测。
 *
 * @param rawTimestampMs 原始时间戳。调用方应当传 `requestVideoFrameCallback` 的
 *   `now`（与 `performance.now()` 同源），**不要传 `metadata.mediaTime` 或
 *   `video.currentTime`** —— 那是另一条时间线，混用会让下面的守卫失去意义。
 */
export function detectFacesFromVideo(
  video: HTMLVideoElement,
  rawTimestampMs: number,
): FaceLandmarks[] {
  if (!videoInstance) throw new Error('视频检测实例尚未加载，请先调用 loadVideoLandmarker()')

  const width = video.videoWidth
  const height = video.videoHeight
  // 元数据还没到时报 0，那样所有像素坐标都会是 0，最后只会得到一句
  // 「瞳距无效」的报错，和真实原因对不上。调用方应等 loadedmetadata。
  if (width <= 0 || height <= 0) {
    throw new Error('视频尺寸还是 0，应等 loadedmetadata 之后再开始检测')
  }

  const raw = Number.isFinite(rawTimestampMs) ? rawTimestampMs : performance.now()
  // 严格递增，见 videoTimestamp 的说明
  videoTimestamp = raw > videoTimestamp ? raw : videoTimestamp + 1

  return toFaceLandmarks(videoInstance.detectForVideo(video, videoTimestamp), width, height)
}

