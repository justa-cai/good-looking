/**
 * 在浏览器里用引擎自己的测量式跑一遍样本，输出每张图每个指标的值。
 *
 * 必须在 src 模块可访问的页面上执行（dev server 或已构建的页面），
 * 所以用 playwright-cli 的 eval 跑：
 *
 *   playwright-cli -s=cal open http://127.0.0.1:5273/good-looking/
 *   playwright-cli -s=cal eval "$(cat scripts/calibration/measure.js)" > tmp/calib_raw.txt
 *
 * 图片由 scripts/calibration/serve.py 提供（需要 CORS 头）。
 * 输出交给 stats.py 统计。
 *
 * 关键：这里调用的就是 src/engine 的 analyze()，不另写一份测量实现 ——
 * 校准与线上必须是同一套代码，否则数字对不上。
 */
(async () => {
  // fetch_sample.py 抓下来的文件清单
  const FILES = await (async () => {
    const res = await fetch('http://127.0.0.1:8099/sample/')
    const html = await res.text()
    return [...html.matchAll(/href="([^"?][^"]*)"/g)]
      .map((m) => m[1])
      .filter((h) => !h.endsWith('/'))
  })()
  if (FILES.length === 0) return JSON.stringify({ error: '样本为空，先跑 fetch_sample.py' })

  const lm = await import('/good-looking/src/face/landmarker.ts')
  const img = await import('/good-looking/src/image.ts')
  const eng = await import('/good-looking/src/engine/index.ts')

  const first = await (await fetch('http://127.0.0.1:8099/sample/' + FILES[0])).blob()
  const warm = await img.prepareImage(first)
  lm.dispose()
  await lm.loadLandmarker({ delegate: 'CPU', probe: warm.bitmap })

  const rows = []
  for (const f of FILES) {
    try {
      const b = await (await fetch('http://127.0.0.1:8099/sample/' + f)).blob()
      const p = await img.prepareImage(b)
      const faces = lm.detectFaces(p.bitmap)
      p.bitmap.close()
      if (faces.length === 0) {
        rows.push({ file: f, error: 'no-face' })
        continue
      }
      const r = eng.analyze(faces[0].pixels)
      const values = {}
      for (const m of r.metrics) values[m.id] = m.value
      rows.push({ file: f, values })
    } catch (e) {
      rows.push({ file: f, error: String((e && e.message) || e) })
    }
  }
  return JSON.stringify(rows)
})()
