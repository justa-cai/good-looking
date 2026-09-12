/**
 * 应用入口。
 *
 * 这一层只负责把各个模块接起来，不放任何推理逻辑：
 *   face/    人脸检测、关键点、姿态、质量校验
 *   engine/  几何打分（纯函数，零依赖）
 *   model/   学习模型推理（路线 B，尚未接入）
 *   ui/      界面渲染
 */

import './style.css'
import { mountApp } from './app.ts'

mountApp()
