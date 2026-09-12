/**
 * 应用入口。
 *
 * 这一层只负责把各个模块接起来，不放任何推理逻辑：
 *   face/    人脸检测与关键点
 *   engine/  几何打分（纯函数，零依赖）
 *   model/   学习模型推理
 *   ui/      界面渲染
 */

import './style.css'

const app = document.querySelector<HTMLDivElement>('#app')
if (!app) throw new Error('#app 容器不存在')

app.innerHTML = `
  <main class="shell">
    <header class="shell__head">
      <h1>good-looking</h1>
      <p class="shell__tagline">浏览器端人脸颜值分析</p>
    </header>

    <section class="shell__body">
      <p class="shell__privacy">
        所有推理在你的设备上完成，照片不会被上传。
      </p>
      <p class="shell__wip">正在构建中。</p>
    </section>
  </main>
`
