import { defineConfig } from 'vite'

/**
 * GitHub Pages 部署在子路径下，base 必须是仓库名。
 * 改了仓库名就要同步改这里，否则构建产物里的资源路径全部 404。
 * 用 `pnpm build && pnpm preview` 验证。
 */
const REPO_NAME = 'good-looking'

export default defineConfig({
  base: `/${REPO_NAME}/`,
  build: {
    target: 'es2022',
    // 模型与 wasm 体积大，别让 Vite 内联成 base64 把 JS 撑爆
    assetsInlineLimit: 4096,
    rollupOptions: {
      output: {
        // 把媒体/推理库拆成独立 chunk：主包保持很小，这两个可以长期缓存
        manualChunks(id) {
          if (id.includes('@mediapipe/tasks-vision')) return 'mediapipe'
          if (id.includes('onnxruntime-web')) return 'ort'
          return undefined
        },
      },
    },
  },
  server: {
    // 注意：CDP 常驻 Chrome 用的是 9333，别撞上
    port: 5273,
    /**
     * GitHub Pages 无法设置响应头，所以生产环境没有 SharedArrayBuffer，
     * 多线程 WASM 不可用。这里也不开 COOP/COEP —— 保证 dev 与线上行为一致，
     * 否则会出现「本地能跑、线上崩」这种最难查的问题。
     */
  },
})
