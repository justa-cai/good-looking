import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { defineConfig, type Plugin } from 'vite'

const pkg = JSON.parse(
  readFileSync(fileURLToPath(new URL('./package.json', import.meta.url)), 'utf8'),
) as { dependencies: Record<string, string> }

/**
 * GitHub Pages 部署在子路径下，base 必须是仓库名。
 * 改了仓库名就要同步改这里，否则构建产物里的资源路径全部 404。
 * 用 `pnpm build && pnpm preview` 验证。
 */
const REPO_NAME = 'good-looking'

/**
 * 把 onnxruntime-web 带进产物里的 wasm 删掉，改由 CDN 提供。
 *
 * 为什么：`onnxruntime-web` 内部用 `new URL('...wasm', import.meta.url)` 引用
 * 自己的 wasm，Vite 看到就会把它当资源拷进产物 —— 那是 **27.8 MB** 的死重量。
 * 而我们在 `classifier.ts` 里已经把 `ort.env.wasm.wasmPaths` 指到了 jsDelivr，
 * 运行时根本不会去读产物里那份（`wasmPaths` 优先于打包时解析出的 URL）。
 *
 * 于是这里只做一件事：产物里的 `ort-wasm*.wasm` 不写出去。
 * 留着它的代价是 —— Pages 站点从 55 MB 涨到 83 MB，而这些字节永远不会被请求。
 *
 * ⚠️ 删掉之后**必须实测运行时能不能从 CDN 加载**（playwright 走一遍路线 B）。
 * 如果 ORT 哪天改成优先用打包时解析的 URL，表现会是「模型下完了但会话建不起来」，
 * 而不是构建报错 —— 这种失败在 CI 里看不见。
 */
function stripOrtWasm(): Plugin {
  return {
    name: 'strip-ort-wasm',
    // ⚠️ 必须是 'post'。wasm 是 Vite 自己的 vite:asset 插件在 generateBundle 里
    // 写出去的，普通插件顺序下我们的钩子跑在它前面，那时 bundle 里还没有这个文件，
    // 循环一圈什么也删不掉（踩过）。enforce: 'post' 保证最后执行。
    enforce: 'post',
    generateBundle(_options, bundle) {
      for (const [name, output] of Object.entries(bundle)) {
        if (/ort-wasm.*\.wasm$/.test(output.fileName)) {
          delete bundle[name]
          console.info(`[strip-ort-wasm] 移出产物：${name}（改由 CDN 提供）`)
        }
      }
    },
  }
}

export default defineConfig({
  base: `/${REPO_NAME}/`,

  plugins: [stripOrtWasm()],

  define: {
    /**
     * MediaPipe 的 JS 由 Vite 从 node_modules 打包，但 wasm 在运行时从 CDN 拉。
     * 两者版本必须严格一致，否则会出现很难定位的加载失败。
     * 这里从 package.json 读，保证永远跟依赖同步，不会写死一个会过期的版本号。
     */
    __MEDIAPIPE_VERSION__: JSON.stringify(pkg.dependencies['@mediapipe/tasks-vision']),
  },

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
