/**
 * 模型文件的获取与缓存。
 *
 * 为什么不用 `InferenceSession.create(url)` 让它自己下载：
 * 那样拿不到**下载进度**，也用不上 **Cache API**。54 MB 的模型在慢网络下
 * 没有任何进度反馈是不可接受的，而每次刷新都重下 54 MB 更不可接受。
 *
 * 缓存用 Cache API 而不是 HTTP 缓存，理由是**可控**：
 * - HTTP 缓存什么时候被清、会不会命中，完全取决于浏览器和响应头；
 * - Cache API 是我们自己命名、自己查、自己删的，行为可预测；
 * - 而且它存的是响应体本身，命中时不发起任何网络请求，离线也能用。
 *
 * ⚠️ Cache API 只在**安全上下文**（https 或 localhost）可用。
 * GitHub Pages 是 https，本地 dev 是 localhost，都没问题；
 * 但如果是通过局域网 IP 用 http 访问（比如手机连电脑调试），`caches` 是 undefined。
 * 那种情况下退化成「每次都重新下载」，**不报错、功能不受影响** —— 只是慢。
 * 不要把它当成致命错误。
 */

/** 缓存名。带版本号，将来换了模型可以直接换名字，老缓存自然淘汰。 */
const CACHE_NAME = 'good-looking-models-v1'

/**
 * 模型体积的合理下限。我们的模型是 54 MB，低于这个数一定是别的什么东西。
 *
 * 存在的理由是**必须挡住「200 但内容不是模型」**：不少静态服务器
 * （Vite dev server、各种 SPA 配置）对不存在的路径会返回 200 + index.html，
 * 而不是 404。这种情况下 `res.ok` 是 true、响应体非空，
 * 光看状态码完全发现不了问题，结果就是把一个 HTML 页面当成 54 MB 的模型
 * 缓存起来，然后 ORT 报一个和真实原因毫无关系的解析错误。
 *
 * ⚠️ 这和 CLAUDE.md 里记的图片那个坑是同一类问题：
 * 「404 返回 HTML 页 → 上层报出一个误导性的解码错误」。
 * **看到解析类报错先查拿到的到底是什么内容，不要先怀疑格式。**
 */
const MIN_MODEL_BYTES = 1_000_000

/**
 * 内容看起来像文本（HTML 错误页 / JSON 错误响应）就判为无效。
 *
 * 判据：跳过开头的空白后第一个字节是 `<`。protobuf 二进制（ONNX 就是）
 * 不会以 `<` 开头，所以这个判断不会误伤真的模型。
 */
function looksLikeText(bytes: ArrayBuffer): boolean {
  const head = new Uint8Array(bytes, 0, Math.min(64, bytes.byteLength))
  for (const b of head) {
    // 空白字符：空格、\t、\n、\r
    if (b === 0x20 || b === 0x09 || b === 0x0a || b === 0x0d) continue
    return b === 0x3c // '<'
  }
  return false
}

/**
 * 拿到的东西到底是不是模型。不是就抛，让调用方去试下一个地址 ——
 * **绝不能把无效内容写进缓存**，否则下次连网络都不会走。
 */
function validateModelBytes(bytes: ArrayBuffer): void {
  if (bytes.byteLength === 0) {
    throw new Error('响应体是空的')
  }
  if (looksLikeText(bytes)) {
    const head = new TextDecoder().decode(bytes.slice(0, 80)).replace(/\s+/g, ' ')
    throw new Error(`返回的是文本而不是模型文件（多半是错误页）：${head}`)
  }
  if (bytes.byteLength < MIN_MODEL_BYTES) {
    throw new Error(
      `响应体只有 ${bytes.byteLength} 字节，远小于模型的预期体积，多半是被截断或指向了别的文件`,
    )
  }
}

export interface FetchProgress {
  readonly received: number
  /** 服务器没给 Content-Length 时为 null */
  readonly total: number | null
}

export interface ModelBytes {
  readonly bytes: ArrayBuffer
  /** 最终成功的那个地址，便于日志里看清楚是从哪来的 */
  readonly url: string
  /** 是否来自缓存（true 表示这次没走网络） */
  readonly fromCache: boolean
}

function cacheAvailable(): boolean {
  return typeof caches !== 'undefined'
}

/** 命中就返回，没有或读失败都返回 null（读失败不该让整个流程挂掉）。 */
async function readCache(url: string): Promise<ArrayBuffer | null> {
  if (!cacheAvailable()) return null
  try {
    const cache = await caches.open(CACHE_NAME)
    const hit = await cache.match(url)
    if (!hit) return null
    return await hit.arrayBuffer()
  } catch (err) {
    console.warn('[model] 读缓存失败，当作没命中：', err)
    return null
  }
}

/**
 * 写缓存。失败只是警告 —— 比如配额不够、或者存不下。
 * 缓存是优化，不是功能，写不进去不该让用户看不到结果。
 */
async function writeCache(url: string, bytes: ArrayBuffer): Promise<void> {
  if (!cacheAvailable()) return
  try {
    const cache = await caches.open(CACHE_NAME)
    await cache.put(url, new Response(bytes.slice(0), {
      headers: {
        'Content-Type': 'application/octet-stream',
        // 记下我们自己写入的时间，排查「缓存是不是旧的」时有用
        'X-Cached-At': new Date().toISOString(),
      },
    }))
  } catch (err) {
    console.warn('[model] 写缓存失败（不影响本次使用）：', err)
  }
}

/**
 * 带进度地读一个响应体。
 *
 * 用手工 reader 循环而不是 `res.arrayBuffer()`：后者要等全部下载完才返回，
 * 中间拿不到任何进度。
 *
 * ⚠️ **响应被压缩时 `Content-Length` 不能当分母。** 这是部署到 GitHub Pages
 * 之后才暴露的一个坑（本地 dev / preview 都不会压缩，所以本地永远看不到）：
 * Pages 对 `.onnx` 会带上 `content-encoding: gzip` 和一个 **52734767**（压缩后）的
 * `Content-Length`，而 `reader.read()` 拿到的每个分片都是**解压后**的字节，
 * 加起来是 **56835454**。两者一比就是「已下载 53.8 MB / 共 50.3 MB」——
 * received 比 total 还大，界面上看起来像坏了。
 *
 * 所以有内容编码时直接不给 total（`null`），界面会退化成只显示已下载多少 MB，
 * 不显示百分比。宁可少一个百分比，也不要摆一个自相矛盾的比例给用户看。
 */
async function readWithProgress(
  res: Response,
  onProgress?: (p: FetchProgress) => void,
): Promise<ArrayBuffer> {
  // Content-Encoding 不在 fetch 的禁止读取响应头名单里，读得到。
  // 'identity' 是「没压缩」的正式写法，和空串一样当作没有编码
  const encoding = (res.headers.get('Content-Encoding') ?? '').trim().toLowerCase()
  const encoded = encoding !== '' && encoding !== 'identity'
  const declared = Number(res.headers.get('Content-Length')) || null
  const total = encoded ? null : declared

  // 没有 body（比如某些代理）时退回一次性读，至少不会挂
  if (!res.body) {
    const buf = await res.arrayBuffer()
    onProgress?.({ received: buf.byteLength, total })
    return buf
  }

  const reader = res.body.getReader()
  const chunks: Uint8Array[] = []
  let received = 0

  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    received += value.byteLength
    // 兜底：即使没有编码标记，只要收到的字节数超过了声明的总长，那个总长就一定
    // 不是这把尺子的刻度（中间还有代理、分块传输等各种情况）。此时撤掉百分比，
    // 而不是继续报一个 100% 以上的数
    onProgress?.({
      received,
      total: total !== null && received <= total ? total : null,
    })
  }

  const out = new Uint8Array(received)
  let offset = 0
  for (const c of chunks) {
    out.set(c, offset)
    offset += c.byteLength
  }
  return out.buffer
}

/** 删掉某个地址的缓存条目。失败只警告 —— 删不掉也不该让流程挂掉。 */
async function evict(url: string): Promise<void> {
  if (!cacheAvailable()) return
  try {
    const cache = await caches.open(CACHE_NAME)
    await cache.delete(url)
  } catch (err) {
    console.warn('[model] 删缓存条目失败：', err)
  }
}

/**
 * 依次尝试候选地址，返回第一个成功的模型字节。
 *
 * 顺序即优先级：排在前面的本地/官方源先试，CDN 兜底。
 * 每个候选都先查缓存 —— 只要之前从**任意一个**候选下载成功过，
 * 换到别的候选时也能命中（缓存键是候选 URL 本身，所以是逐个查的）。
 */
export async function fetchModelBytes(
  urls: readonly string[],
  opts: { onProgress?: (p: FetchProgress) => void; signal?: AbortSignal } = {},
): Promise<ModelBytes> {
  if (urls.length === 0) throw new Error('没有配置任何模型地址')

  const failures: string[] = []

  for (const url of urls) {
    const cached = await readCache(url)
    if (cached) {
      // 缓存命中也要验一遍：这份缓存可能是旧版本代码写进去的（那时还没有校验），
      // 也可能是硬盘/配额问题导致的半截文件。验不过就删掉它、继续走网络 ——
      // 让缓存能自愈，而不是把用户永久卡在一个坏缓存上。
      try {
        validateModelBytes(cached)
        // 缓存命中也要报一次进度，否则界面会一直停在 0%
        opts.onProgress?.({ received: cached.byteLength, total: cached.byteLength })
        return { bytes: cached, url, fromCache: true }
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err)
        console.warn(`[model] 缓存里那份不可用（${msg}），删掉重新下载`)
        await evict(url)
      }
    }

    try {
      // signal 必须按需带上：exactOptionalPropertyTypes 下不能传 undefined，
      // 而 RequestInit.signal 的类型是 AbortSignal | null，塞 undefined 也不合法
      const init: RequestInit = opts.signal ? { signal: opts.signal } : {}
      const res = await fetch(url, init)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)

      const bytes = await readWithProgress(res, opts.onProgress)
      // 先验内容再进缓存：不验的话，一个 200 的错误页会被当成模型存下来，
      // 之后连网络都不会再走，问题被永久固化
      validateModelBytes(bytes)

      await writeCache(url, bytes)
      return { bytes, url, fromCache: false }
    } catch (err) {
      // AbortError 是用户主动取消，不该继续试下一个源
      if (err instanceof DOMException && err.name === 'AbortError') throw err
      const msg = err instanceof Error ? err.message : String(err)
      console.warn(`[model] ${url} 取不到（${msg}），试下一个`)
      failures.push(`${url} → ${msg}`)
    }
  }

  throw new Error(`所有模型地址都取不到：\n${failures.join('\n')}`)
}

/**
 * 清掉模型缓存。用于「换个模型试试」或者排查「是不是缓存坏了」。
 * 界面上不必暴露，但排查问题时能省掉「手动去 DevTools 里删」这一步。
 */
export async function clearModelCache(): Promise<void> {
  if (!cacheAvailable()) return
  await caches.delete(CACHE_NAME)
}

/** 缓存里已经存了哪些模型地址。排查用。 */
export async function listCachedModels(): Promise<string[]> {
  if (!cacheAvailable()) return []
  const cache = await caches.open(CACHE_NAME)
  return (await cache.keys()).map((r) => r.url)
}
