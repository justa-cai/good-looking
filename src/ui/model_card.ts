/**
 * 路线 B（学习模型）的结果卡片。
 *
 * ⚠️ 这张卡片的设计**完全由实测结论决定**，改之前先读 CLAUDE.md
 * 「绝对概率不能当分数用」那一节：
 *
 * 这个模型在 37 张国会议员官方肖像（全部是清晰正脸的专业证件照）上的
 * P(attractive) 中位数只有 **0.12**，均值 0.23，只有 7/37 超过 0.5。
 * 直接把它线性映射成 0-100 分，等于给所有人一个稳定的低分，
 * 而且低的原因和脸无关（训练分布是名人颜值图）。
 *
 * 所以这里**不出现任何「分」**，只做三件事：
 *   1. 如实报告模型输出的是哪一类、概率是多少；
 *   2. 把参照分布（那 37 张的中位数）摆出来，让数字有解读的锚点；
 *   3. 明说它不能当分数用。
 *
 * 和路线 A 的分数**并排但不合并** —— 两者量的不是一回事，
 * 合成一个数会让「哪个数字是谁说了算」变得不可解释。
 */

import type { Classification } from '../model/classifier.ts'

/** 参照分布。来自 `scripts/model/score_corpus.py` 的实测输出，改模型要重跑。 */
const REFERENCE = {
  /** 样本数 */
  n: 37,
  /** P(attractive) 的中位数 */
  median: 0.123,
  mean: 0.231,
  /** 高于这个值的张数 */
  above: 7,
  /** 样本是什么 */
  what: '美国国会议员官方肖像照',
} as const

export type ModelCardState =
  | { readonly kind: 'idle' }
  | { readonly kind: 'loading'; readonly received: number; readonly total: number | null }
  | { readonly kind: 'running' }
  | { readonly kind: 'done'; readonly result: Classification }
  | { readonly kind: 'error'; readonly message: string }

export interface ModelCardHandlers {
  /** 用户点了「加载并分析」 */
  readonly onStart: () => void
}

function fmtMB(bytes: number): string {
  return (bytes / 1024 / 1024).toFixed(1)
}

export function renderModelCard(
  panel: HTMLElement,
  state: ModelCardState,
  handlers: ModelCardHandlers,
  /** 模型已下载过（缓存命中或本次会话已加载），不必再问用户 */
  known: boolean,
): void {
  panel.replaceChildren()

  const card = document.createElement('section')
  card.className = 'model-card'
  card.setAttribute('aria-label', '学习模型结果')

  const head = document.createElement('div')
  head.className = 'model-card__head'
  head.innerHTML = `<h2>学习模型（路线 B）</h2>`
  card.append(head)

  const body = document.createElement('div')
  body.className = 'model-card__body'

  switch (state.kind) {
    case 'idle': {
      const p = document.createElement('p')
      p.className = 'model-card__note'
      p.textContent = known
        ? '模型已经就绪，正在分析。'
        : `模型 54 MB，只需要下载一次，之后从本地缓存读取。这一步是可选的，上面的比例分析不受影响。`
      body.append(p)

      if (!known) {
        const btn = document.createElement('button')
        btn.type = 'button'
        btn.className = 'model-card__btn'
        btn.textContent = '下载并分析'
        btn.addEventListener('click', handlers.onStart)
        body.append(btn)
      }
      break
    }

    case 'loading': {
      const pct =
        state.total && state.total > 0
          ? Math.min(100, Math.round((state.received / state.total) * 100))
          : null
      const wrap = document.createElement('div')
      wrap.className = 'model-card__progress'
      wrap.innerHTML = `
        <div class="model-card__bar"><div class="model-card__fill" style="width:${pct ?? 8}%"></div></div>
        <p class="model-card__note">${
          pct === null
            ? `正在下载模型… ${fmtMB(state.received)} MB`
            : `正在下载模型… ${pct}%（${fmtMB(state.received)} / ${fmtMB(state.total!)} MB）`
        }</p>
      `
      body.append(wrap)
      break
    }

    case 'running': {
      const p = document.createElement('p')
      p.className = 'model-card__note'
      p.innerHTML = `<span class="status__spinner"></span>正在推理…`
      body.append(p)
      break
    }

    case 'done': {
      const { pAttractive } = state.result
      const pct = (pAttractive * 100).toFixed(1)
      const positive = pAttractive >= 0.5
      const verdict = positive ? 'attractive' : 'not attractive'

      const row = document.createElement('div')
      row.className = 'model-card__verdict'
      row.innerHTML = `
        <span class="model-card__class">${verdict}</span>
        <span class="model-card__prob">P(attractive) = ${pct}%</span>
      `
      body.append(row)

      // 参照锚点：不给参照，用户只会看到「12%」然后以为自己做错了什么
      const ref = document.createElement('p')
      ref.className = 'model-card__ref'
      ref.textContent =
        `参照：${REFERENCE.n} 张${REFERENCE.what}在这个模型上的中位数是 ` +
        `${(REFERENCE.median * 100).toFixed(1)}%，只有 ${REFERENCE.above} 张超过 50%。`
      body.append(ref)

      // 这是这张卡片存在的理由，不能省
      const caveat = document.createElement('p')
      caveat.className = 'model-card__caveat'
      caveat.textContent =
        '这个数字不是「颜值分」，也请不要和上面的 /100 对比 —— ' +
        '两者量的不是一回事。模型的训练数据是名人颜值图，和人像证件照的分布差得很远，' +
        '导致它对大多数人给出偏低的值。它相对稳定（同一人不同照片差异很小），' +
        '但绝对高低和长相好坏没有可靠对应。'
      body.append(caveat)
      break
    }

    case 'error': {
      const p = document.createElement('p')
      p.className = 'model-card__error'
      p.textContent = `模型这一步没做成：${state.message}`
      const hint = document.createElement('p')
      hint.className = 'model-card__note'
      hint.textContent = '上面的比例分析不受影响。可以重试一次，或者先只看比例分析。'
      const btn = document.createElement('button')
      btn.type = 'button'
      btn.className = 'model-card__btn'
      btn.textContent = '重试'
      btn.addEventListener('click', handlers.onStart)
      body.append(p, hint, btn)
      break
    }
  }

  card.append(body)
  panel.append(card)
}
