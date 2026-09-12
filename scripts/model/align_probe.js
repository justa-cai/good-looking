/**
 * 验证人脸对齐确实有用（`src/face/align.ts`）。
 *
 * CLAUDE.md 里「对齐让模型对旋转几乎不敏感」和「对齐把同一人不同照片的
 * 极差降到 1/3」两张表就是这个脚本跑出来的。改了 align.ts 的任何参数
 * （CROP_IPD_RATIO、EYE_LINE_Y）都要重跑，否则那两张表就过期了。
 *
 * 用法：
 *   pnpm dev &
 *   python3 scripts/calibration/serve.py 8099 tmp/faces &
 *   # 模型要先放到能被 8099 访问的位置
 *   cp tmp/models/attractive.int4.onnx tmp/faces/models/
 *   playwright-cli -s=cal open http://127.0.0.1:5273/good-looking/
 *   playwright-cli -s=cal eval "$(cat scripts/model/align_probe.js)" > tmp/align_probe.txt
 *
 * 输出两段：
 *   plain  —— 每张图的「整图拉伸」与「对齐裁剪」两种输入的 P(attractive)
 *   rotate —— 抽 3 张，各转 -20/-10/0/+10/+20 度后重新对齐再推理，
 *             看极差有多小（对齐有效的话应该很小）
 *
 * ⚠️ 模型走 CDN 导入 ORT。这里不 import 项目里的 onnxruntime-web，
 * 因为这条脚本要能在只有 dev server 的情况下跑，不想为一个校验脚本
 * 把 ORT 拉进主 bundle。
 */
(async () => {
  const ort = await import('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/ort.min.mjs')
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.29.0/dist/'

  const lm = await import('/good-looking/src/face/landmarker.ts')
  const img = await import('/good-looking/src/image.ts')
  const q = await import('/good-looking/src/face/quality.ts')
  const al = await import('/good-looking/src/face/align.ts')

  const MODEL = 'http://127.0.0.1:8099/models/attractive.int4.onnx'
  const SAMPLE = 'http://127.0.0.1:8099/sample/'

  const res = await fetch(SAMPLE)
  const html = await res.text()
  const FILES = [...html.matchAll(/href="([^"?][^"]*)"/g)]
    .map((m) => m[1])
    .filter((h) => !h.endsWith('/'))
  if (FILES.length === 0) return JSON.stringify({ error: '样本为空，先跑 fetch_sample.py' })

  const buf = await (await fetch(MODEL)).arrayBuffer()
  const sess = await ort.InferenceSession.create(buf, { executionProviders: ['wasm'] })

  const warm = await img.prepareImage(await (await fetch(SAMPLE + FILES[0])).blob())
  lm.dispose()
  // 固定 CPU：不同后端的关键点有细微差别，校验要可复现
  await lm.loadLandmarker({ delegate: 'CPU', probe: warm.bitmap })

  /** 位图 → 模型输入张量。ViT 的归一化：x/127.5 - 1（即 (x/255-0.5)/0.5） */
  function toTensor(bitmap) {
    const c = new OffscreenCanvas(224, 224)
    const ctx = c.getContext('2d')
    ctx.drawImage(bitmap, 0, 0, 224, 224)
    const d = ctx.getImageData(0, 0, 224, 224).data
    const out = new Float32Array(3 * 224 * 224)
    for (let i = 0, p = 0; i < 224 * 224; i++, p += 4) {
      out[i] = d[p] / 127.5 - 1
      out[224 * 224 + i] = d[p + 1] / 127.5 - 1
      out[2 * 224 * 224 + i] = d[p + 2] / 127.5 - 1
    }
    return new ort.Tensor('float32', out, [1, 3, 224, 224])
  }

  async function predict(bitmap) {
    const r = await sess.run({ pixel_values: toTensor(bitmap) })
    const l = Array.from(r.logits.data)
    const m = Math.max(l[0], l[1])
    const e = [Math.exp(l[0] - m), Math.exp(l[1] - m)]
    return e[0] / (e[0] + e[1])
  }

  async function rotate(bitmap, deg) {
    if (deg === 0) return bitmap
    const c = new OffscreenCanvas(bitmap.width, bitmap.height)
    const ctx = c.getContext('2d')
    ctx.translate(bitmap.width / 2, bitmap.height / 2)
    ctx.rotate((deg * Math.PI) / 180)
    ctx.translate(-bitmap.width / 2, -bitmap.height / 2)
    ctx.drawImage(bitmap, 0, 0)
    return createImageBitmap(c)
  }

  const out = { plain: [], rotate: [] }

  for (const f of FILES.slice(0, 8)) {
    const p = await img.prepareImage(await (await fetch(SAMPLE + f)).blob())
    const rep = q.checkQuality(lm.detectFaces(p.bitmap))
    if (!rep.face) {
      out.plain.push({ f: f.slice(0, 30), skip: rep.issues.map((i) => i.code) })
      p.bitmap.close()
      continue
    }
    const a = await al.alignFace(p.bitmap, rep.face)
    if (!a) {
      out.plain.push({ f: f.slice(0, 30), skip: ['align-failed'] })
      p.bitmap.close()
      continue
    }
    out.plain.push({
      f: f.slice(0, 30),
      naive: +((await predict(p.bitmap)).toFixed(4)),
      aligned: +((await predict(a.bitmap)).toFixed(4)),
      roll: +a.rollDeg.toFixed(1),
    })
    a.bitmap.close()
    p.bitmap.close()
  }

  for (const f of FILES.slice(0, 3)) {
    const p = await img.prepareImage(await (await fetch(SAMPLE + f)).blob())
    const row = { f: f.slice(0, 30) }
    for (const deg of [-20, -10, 0, 10, 20]) {
      const rb = await rotate(p.bitmap, deg)
      const rep = q.checkQuality(lm.detectFaces(rb))
      let v = null
      if (rep.face) {
        const a = await al.alignFace(rb, rep.face)
        if (a) {
          v = +((await predict(a.bitmap)).toFixed(4))
          a.bitmap.close()
        }
      }
      row['d' + deg] = v
      if (rb !== p.bitmap) rb.close()
    }
    out.rotate.push(row)
    p.bitmap.close()
  }

  return JSON.stringify(out)
})()
