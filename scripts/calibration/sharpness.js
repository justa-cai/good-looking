/**
 * 清晰度阈值的校准。
 *
 * `src/face/quality.ts` 里的 minSharpness / warnSharpness 不是手填的，
 * 是用 `src/face/sharpness.ts` 的度量在一批照片上跑出来、再合成模糊样本对比得到的。
 *
 * 做法：拿同一批「清晰」样本（官方肖像），对**缩放后的位图**施加浏览器自己的
 * CSS `blur(Npx)`，模拟「用户上传了一张糊照片」。两个分布一分，
 * 就能看出阈值该放哪、以及这一关的能力边界在哪。
 *
 * ⚠️ 必须用 CSS 滤镜，不要用 PIL。PIL 的 GaussianBlur(radius) 里 radius 就是 σ，
 * 和 CSS blur 的标定对不上（PIL σ=1.0 实测 10.05，CSS blur(2px) 实测约 2-4），
 * 差 3 倍以上，用错工具会把阈值调到错的地方。
 *
 * ⚠️ 模糊要加在**缩放之后**的位图上。先糊原图再缩小，模糊半径会被缩放吃掉，
 * 严重低估糊的程度（第一版就踩了这个坑，blur(4px) 的中位还在 0.6 以上）。
 *
 * 用法：
 *   pnpm dev &
 *   python3 scripts/calibration/serve.py 8099 tmp/faces &
 *   playwright-cli -s=cal open http://127.0.0.1:5273/good-looking/
 *   playwright-cli -s=cal eval "$(cat scripts/calibration/sharpness.js)" > tmp/sharpness_raw.txt
 *   python3 scripts/calibration/sharpness_stats.py tmp/sharpness_raw.txt
 */

(async () => {
  const res = await fetch('http://127.0.0.1:8099/sample/')
  const html = await res.text()
  const FILES = [...html.matchAll(/href="([^"?][^"]*)"/g)]
    .map((m) => m[1])
    .filter((h) => !h.endsWith('/'))
  if (FILES.length === 0) return JSON.stringify({ error: '样本为空，先跑 fetch_sample.py' })

  const lm = await import('/good-looking/src/face/landmarker.ts')
  const img = await import('/good-looking/src/image.ts')
  const sh = await import('/good-looking/src/face/sharpness.ts')
  const q = await import('/good-looking/src/face/quality.ts')

  /** 要合成的模糊半径（CSS blur(Npx)）。0 是清晰对照组。 */
  const LEVELS = [0, 1, 2, 3, 4]

  async function blurBitmap(bmp, radius) {
    if (radius === 0) return bmp
    const c = document.createElement('canvas')
    c.width = bmp.width
    c.height = bmp.height
    const ctx = c.getContext('2d')
    ctx.filter = `blur(${radius}px)`
    ctx.drawImage(bmp, 0, 0)
    return createImageBitmap(c)
  }

  const first = await (await fetch('http://127.0.0.1:8099/sample/' + FILES[0])).blob()
  const warm = await img.prepareImage(first)
  lm.dispose()
  // 固定用 CPU：不同后端的检测框会有细微差别，校准要可复现
  await lm.loadLandmarker({ delegate: 'CPU', probe: warm.bitmap })

  const rows = []
  for (const f of FILES) {
    const blob = await (await fetch('http://127.0.0.1:8099/sample/' + f)).blob()
    const base = await img.prepareImage(blob)
    const row = { f, w: base.width, h: base.height }
    for (const r of LEVELS) {
      const bmp = await blurBitmap(base.bitmap, r)
      const faces = lm.detectFaces(bmp)
      const rep = q.checkQuality(faces)
      if (!rep.face) {
        row['b' + r] = null
      } else {
        const m = sh.measureSharpness(bmp, rep.face)
        row['b' + r] = m ? Math.round(m.sharpness * 100) / 100 : null
        row['raw' + r] = m ? Math.round(m.rawVariance) : null
        row['sd' + r] = m ? Math.round(m.lumaSd) : null
      }
      if (bmp !== base.bitmap) bmp.close()
    }
    base.bitmap.close()
    rows.push(row)
  }
  return JSON.stringify(rows)
})()
