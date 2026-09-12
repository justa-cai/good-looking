/**
 * MediaPipe FaceMesh 关键点索引。
 *
 * 取值来自 Google 官方 face_mesh_connections.py（Apache-2.0）里的连接定义，
 * 不是凭印象写的。改名或新增索引前请回官方定义核对：
 * https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py
 *
 * ⚠️ 关于左右与 A/B 后缀
 *
 * MediaPipe 的 LEFT_/RIGHT_ 是**被摄者的**左右，不是画面上的左右。正脸照里
 * 被摄者的右眼出现在画面左侧；自拍镜像后又会反过来。所以本文件一律不用
 * left/right 命名，改用中性的 A/B 后缀。
 *
 * 规律（官方定义如此，可直接核对）：
 *   眼睛/眉毛/鼻翼/嘴角 —— A 是被摄者右侧的点，B 是左侧的点。
 *   面部轮廓 —— 例外：A=234 是被摄者右侧，B=454 是左侧，但两者在
 *   轮廓上的位置对称，语义与上面一致。
 *
 * 画面上的实际左右由 `pickSides()` 按坐标判断，不要靠这里的命名推断。
 * 需要 A 恒等于「画面左侧」的地方，务必先过一遍 pickSides()。
 */
export const LM = {
  // ---- 面部轮廓（FACEMESH_FACE_OVAL）----
  /** 发际线正中，脸的最上缘 */
  foreheadTop: 10,
  /** 下巴最低点 */
  chin: 152,
  /** 轮廓最外缘，大致在颧骨高度。A=被摄者右，B=被摄者左。 */
  contourA: 234,
  contourB: 454,
  /** 下颌角附近，用于下颌宽度 */
  jawA: 172,
  jawB: 397,

  // ---- 眼睛 ----
  /** 外眼角 */
  eyeOuterA: 33,
  eyeOuterB: 263,
  /** 内眼角 */
  eyeInnerA: 133,
  eyeInnerB: 362,
  /** 上眼睑最高点，用于眼裂高度 */
  eyelidUpperA: 159,
  eyelidUpperB: 386,
  /** 下眼睑最低点 */
  eyelidLowerA: 145,
  eyelidLowerB: 374,

  // ---- 虹膜（478 点模型才有；468-472 一侧，473-477 另一侧）----
  /** 瞳孔中心，比眼角稳定，用于姿态估计 */
  irisA: 468,
  irisB: 473,

  // ---- 眉毛 ----
  browOuterA: 105,
  browOuterB: 334,

  // ---- 鼻子 ----
  /** 鼻根（山根起点）—— 上庭与中庭的分界 */
  nasion: 168,
  /** 鼻尖 */
  noseTip: 1,
  /** 鼻底中点（鼻下点）—— 中庭与下庭的分界 */
  subnasale: 2,
  /** 鼻翼最外点，用于鼻宽 */
  alaA: 48,
  alaB: 278,

  // ---- 嘴唇 ----
  /** 嘴角 */
  mouthCornerA: 61,
  mouthCornerB: 291,
  /** 上唇上缘最高点 */
  lipTop: 0,
  /** 下唇下缘最低点 */
  lipBottom: 17,
  /** 唇内缝上下缘，用于上下唇厚度 */
  lipInnerUpper: 13,
  lipInnerLower: 14,
} as const

/** 关键点总数：带虹膜的 478 点模型 */
export const LANDMARK_COUNT = 478
