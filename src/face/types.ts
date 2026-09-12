/** 图像空间中的点。z 是 MediaPipe 给出的相对深度（越小越靠近镜头），量纲与 x 同。 */
export interface Point3 {
  x: number
  y: number
  z: number
}

/** 一张已检测到人脸的关键点，同时提供归一化与像素两套坐标。 */
export interface FaceLandmarks {
  /**
   * 归一化坐标（0-1），相对原图。MediaPipe 的原始输出，不保证等比。
   * 只用来看比例关系时够用，但做欧氏距离要小心：x 和 y 的量纲不同。
   */
  readonly normalized: readonly Point3[]
  /** 像素坐标，用原图宽高换算过。做几何计算请用这一套。 */
  readonly pixels: readonly Point3[]
  /** 关键点所基于的图像尺寸（已经过下采样，不是用户原图尺寸）。 */
  readonly width: number
  readonly height: number
}
