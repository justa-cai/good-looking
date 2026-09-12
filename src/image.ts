/**
 * 图像加载与降采样。
 *
 * 手机拍的照片动辄 4000×3000，直接喂给 MediaPipe 会明显卡顿，
 * 而且降采样后再推理对关键点精度几乎没有影响。所以统一先缩到 MAX_DIM 以内。
 */

/**
 * 推理用的最长边上限。
 *
 * 取 1280 而不是更小：实测（本机 headless，CPU 后端）从 1025px 降到 256px
 * 只快约 10ms（42ms → 32ms），因为模型内部有自己的输入尺寸，缩放对总耗时
 * 影响很小；但三庭五眼这类指标依赖眼角、嘴角的细节定位，图太小会让关键点
 * 抖动，反而毁掉确定性。所以宁可留大一点。
 */
export const MAX_DIM = 1280

export interface PreparedImage {
  readonly bitmap: ImageBitmap
  readonly width: number
  readonly height: number
  /** 相对原图的缩放比，用于把关键点映射回原图坐标 */
  readonly scale: number
}

function fitWithin(width: number, height: number, maxDim: number): [number, number] {
  const longest = Math.max(width, height)
  if (longest <= maxDim) return [width, height]
  const ratio = maxDim / longest
  return [Math.round(width * ratio), Math.round(height * ratio)]
}

/**
 * 把任意图片文件/Blob 解码成 ImageBitmap，并按需降采样。
 * 用 ImageBitmap 而不是 HTMLImageElement：它能直接喂给 MediaPipe 与 canvas，
 * 且解码在主线程之外完成。
 */
export async function prepareImage(source: Blob, maxDim = MAX_DIM): Promise<PreparedImage> {
  // 先不带缩放解码一次，拿到原始尺寸
  const probe = await createImageBitmap(source)
  const originalWidth = probe.width
  const originalHeight = probe.height

  const [width, height] = fitWithin(originalWidth, originalHeight, maxDim)

  if (width === originalWidth && height === originalHeight) {
    return { bitmap: probe, width, height, scale: 1 }
  }

  // 缩放在解码阶段完成，比先解码再 drawImage 省一次全尺寸位图
  const scaled = await createImageBitmap(source, {
    resizeWidth: width,
    resizeHeight: height,
    resizeQuality: 'high',
  })
  probe.close()

  return { bitmap: scaled, width, height, scale: width / originalWidth }
}

/**
 * 抓视频的一帧，走和照片**完全相同**的处理（同样的降采样上限、同样的坐标语义）。
 *
 * 两条路径必须一致，否则同一个取景下「实时」和「冻结」会给出不同的判定 ——
 * 而冻结的全部意义就是「把此刻看到的那一帧按照片的方式仔细算一遍」。
 *
 * 与 `prepareImage` 的区别是不做「先解码探尺寸」那一步：`videoWidth` 就是原始
 * 尺寸，而且对视频两次 `createImageBitmap` 可能抓到**不同的两帧**。
 */
export async function prepareVideoFrame(
  video: HTMLVideoElement,
  maxDim = MAX_DIM,
): Promise<PreparedImage> {
  const originalWidth = video.videoWidth
  const originalHeight = video.videoHeight
  if (originalWidth <= 0 || originalHeight <= 0) {
    // 元数据还没到。调用方应当先等 loadedmetadata —— 否则所有像素坐标都会是 0，
    // 最后只会得到一句和真实原因对不上的「瞳距无效」
    throw new Error('视频尺寸还是 0，无法抓帧')
  }

  const [width, height] = fitWithin(originalWidth, originalHeight, maxDim)

  // 1920×1080 的摄像头会被这里降到 1280×720，和上传路径同一个上限
  const bitmap =
    width === originalWidth && height === originalHeight
      ? await createImageBitmap(video)
      : await createImageBitmap(video, {
          resizeWidth: width,
          resizeHeight: height,
          resizeQuality: 'high',
        })

  return { bitmap, width, height, scale: width / originalWidth }
}

