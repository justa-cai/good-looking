/**
 * 由 vite.config.ts 的 `define` 注入：当前依赖的 @mediapipe/tasks-vision 版本。
 * wasm 从 CDN 按这个版本号拉取，必须与打包进 bundle 的 JS 版本一致。
 */
declare const __MEDIAPIPE_VERSION__: string
