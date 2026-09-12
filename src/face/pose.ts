/**
 * 从 MediaPipe 的「标准人脸 → 检出人脸」变换矩阵里读头部姿态。
 *
 * 为什么要用它而不是自己拿关键点推：试过按左右关键点的对称性当判据，
 * 实测噪声很大（同一人不同照片的对称性差异比不同人之间还大），
 * 因为关键点本身会随表情和光照漂移。模型直接给出的姿态是基于 3D 人脸模型的
 * 配准结果，稳定得多，也不必自己发明阈值。
 *
 * ============================ 矩阵约定 ============================
 * MediaPipe 给的是 4×4、**行主序**，且是行向量约定（p' = p·M），
 * 因为平移分量落在最后一行（第 4 行前三个）。
 * 所以真正的旋转部分是左上 3×3，且它作用在**右侧**。
 *
 * 提取欧拉角时先转置成教科书里的列向量约定，再按
 * M = Rz(roll) · Ry(yaw) · Rx(pitch) 分解。
 *
 * 角度含义（都是「头相对相机」）：
 *   yaw   左右转头，正 = 转向其左侧（画面上看向右）
 *   pitch 上下点头，正 = 抬头
 *   roll  画面内歪头，正 = 顺时针（与 CSS `rotate()` 的正方向一致）
 *
 * 符号只在相对比较时有意义 —— 判据用的是绝对值，不依赖符号约定。
 * roll 的符号已用受控实验验证：把图按 CSS `rotate(+θ)` 转，读数 ≈ +θ。
 */

export interface HeadPose {
  /** 左右转头，度 */
  readonly yaw: number
  /** 上下点头，度 */
  readonly pitch: number
  /** 画面内歪头，度 */
  readonly roll: number
}

/** 行主序 4×4 矩阵（MediaPipe 的 Matrix.data 顺序）。 */
export type Matrix4 = readonly number[]

const DEG = 180 / Math.PI

/**
 * 取出旋转部分并分解成欧拉角。
 * 输入不是合法的旋转矩阵时返回 null（例如矩阵缺失或全零）。
 */
export function readHeadPose(transform: Matrix4 | null): HeadPose | null {
  if (!transform || transform.length < 16) return null

  // 行主序：R[row][col] = transform[row * 4 + col]
  const r = (row: number, col: number): number => transform[row * 4 + col]!

  // 转置成列向量约定：m = R^T
  const m = (row: number, col: number): number => r(col, row)

  const m00 = m(0, 0)
  const m10 = m(1, 0)
  const m20 = m(2, 0)
  const m21 = m(2, 1)
  const m22 = m(2, 2)

  // 万向节锁（俯仰 ±90°）时 yaw/roll 不可分离，这里用 0 兜底；
  // 正常证件照永远不会到那个角度。
  const yawArg = -m20
  if (yawArg < -1 || yawArg > 1) return null

  const yaw = Math.asin(yawArg) * DEG
  const pitch = Math.atan2(m21, m22) * DEG
  // 取负：矩阵给出的旋转方向与画面上的视觉方向相反，
  // 受控实验（把图按 CSS rotate(+θ) 转）确认取负后 roll 读数 ≈ +θ。
  const roll = -Math.atan2(m10, m00) * DEG

  if (!Number.isFinite(yaw) || !Number.isFinite(pitch) || !Number.isFinite(roll)) return null
  return { yaw, pitch, roll }
}

/** 姿态是否在允许范围内（各自绝对角不超过阈值）。 */
export function isPoseAcceptable(pose: HeadPose, maxAngle: number): boolean {
  return (
    Math.abs(pose.yaw) <= maxAngle &&
    Math.abs(pose.pitch) <= maxAngle &&
    Math.abs(pose.roll) <= maxAngle
  )
}
