/**
 * 跑一遍 `src/model/classifier.ts`，输出每张样本的概率。
 *
 * 两个用途：
 *   1. 确认封装本身能跑通、后端回退正常；
 *   2. 把结果交给 `verify_js_vs_py.py`，和 Python 端的 AutoImageProcessor
 *      比对 —— 这是检查 JS 预处理有没有写错的关键一步。
 *
 * 用法：
 *   pnpm dev &
 *   python3 scripts/calibration/serve.py 8099 tmp/faces &
 *   cp tmp/models/attractive.int4.onnx tmp/faces/models/
 *   playwright-cli -s=cal open http://127.0.0.1:5273/good-looking/
 *   playwright-cli -s=cal eval "$(cat scripts/model/classifier_probe.js)" > tmp/classifier_probe.json
 *   tmp/venv-model/bin/python scripts/model/score_corpus.py --json tmp/model_scores.json
 *   python3 scripts/model/verify_js_vs_py.py tmp/classifier_probe.json tmp/model_scores.json
 *
 * ⚠️ 输出的 js_raw 是**整图直接拉伸**的结果，不做对齐 ——
 * 只有这样才能和 Python 端「整图过 AutoImageProcessor」对上。
 * 要对齐后的数看 align_probe.js。
 */
(async () => {
  const img = await import('/good-looking/src/image.ts')
  const lm = await import('/good-looking/src/face/landmarker.ts')
  const q = await import('/good-looking/src/face/quality.ts')
  const al = await import('/good-looking/src/face/align.ts')
  const cl = await import('/good-looking/src/model/classifier.ts')

  const SAMPLE = 'http://127.0.0.1:8099/sample/'
  const MODEL = 'http://127.0.0.1:8099/models/attractive.int4.onnx'

  const res = await fetch(SAMPLE)
  const html = await res.text()
  const FILES = [...html.matchAll(/href="([^"?][^"]*)"/g)]
    .map((m) => m[1])
    .filter((h) => !h.endsWith('/'))
  if (FILES.length === 0) return JSON.stringify({ error: '样本为空，先跑 fetch_sample.py' })

  const t0 = performance.now()
  await cl.loadClassifier({
    urls: [MODEL],
    onStage: (s) => console.info('[classifier] stage', s),
    onProviderChosen: (p, ms) => console.info('[classifier] provider', p, ms),
  })
  const loadMs = performance.now() - t0

  // 关键点固定 CPU，和校准脚本一致（不同后端的关键点有细微差别）
  const warm = await img.prepareImage(await (await fetch(SAMPLE + FILES[0])).blob())
  lm.dispose()
  await lm.loadLandmarker({ delegate: 'CPU', probe: warm.bitmap })

  const rows = []
  for (const f of FILES) {
    const p = await img.prepareImage(await (await fetch(SAMPLE + f)).blob())
    const rep = q.checkQuality(lm.detectFaces(p.bitmap))

    const raw = await cl.classify(p.bitmap)

    let aligned = null
    if (rep.face) {
      const a = await al.alignFace(p.bitmap, rep.face)
      if (a) {
        aligned = (await cl.classify(a.bitmap)).pAttractive
        a.bitmap.close()
      }
    }

    rows.push({
      file: f,
      js_raw: raw.pAttractive,
      js_aligned: aligned,
      js_margin: raw.logits[0] - raw.logits[1],
    })
    p.bitmap.close()
  }

  return JSON.stringify({
    provider: cl.getActiveProvider(),
    msPerRun: cl.getActiveMsPerRun(),
    loadMs: Math.round(loadMs),
    rows,
  })
})()
