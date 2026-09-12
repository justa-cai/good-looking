/**
 * 结果面板的渲染。
 *
 * 纯 DOM 操作，没有框架 —— 界面元素很少，引框架的收益抵不过依赖成本。
 * 所有数字都来自 AnalysisResult，这一层不做任何计算。
 */

import type { AnalysisResult, MetricResult } from '../engine/index.ts'
import type { Reliability } from '../engine/types.ts'
import { scoreColor } from './overlay.ts'

const IDEAL_SOURCE_LABEL: Record<MetricResult['idealSource'], string> = {
  canon: '经典比例',
  empirical: '人群平均',
  typical: '人群平均',
}

const RELIABILITY_LABEL: Record<Reliability, string | null> = {
  high: null,
  medium: '该项波动较大',
  low: '该项波动很大',
}

function fmt(value: number, digits = 3): string {
  return Number.isFinite(value) ? value.toFixed(digits) : '—'
}

function scoreText(score: number): string {
  return Number.isFinite(score) ? score.toFixed(0) : '—'
}

/** 一句话描述偏离方向，帮助用户理解分数的含义。 */
function deviationText(metric: MetricResult): string {
  if (!Number.isFinite(metric.deviation)) return ''
  const z = metric.deviation
  if (Math.abs(z) < 0.35) return '接近理想值'
  const direction = z > 0 ? '高于' : '低于'
  return `${direction}理想值 ${Math.abs(z).toFixed(1)} 个标准差`
}

function renderMetric(metric: MetricResult): HTMLElement {
  const el = document.createElement('div')
  el.className = 'metric'

  const color = scoreColor(metric.score)
  const reliabilityNote = RELIABILITY_LABEL[metric.reliability]

  const tags: string[] = [`<span class="tag">理想值来源：${IDEAL_SOURCE_LABEL[metric.idealSource]}</span>`]
  if (reliabilityNote) tags.push(`<span class="tag tag--low">${reliabilityNote}</span>`)

  el.innerHTML = `
    <div class="metric__top">
      <span class="metric__label">${metric.label}</span>
      <span class="metric__score" style="color:${color}">${scoreText(metric.score)}</span>
    </div>
    <div class="metric__bar">
      <div class="metric__fill" style="width:${Number.isFinite(metric.score) ? metric.score : 0}%;background:${color}"></div>
    </div>
    <p class="metric__desc">${metric.description}</p>
    <p class="metric__nums">
      实测 ${fmt(metric.value)} · 理想 ${fmt(metric.ideal)} ·
      人群平均 ${fmt(metric.populationMean)}（±${fmt(metric.populationSd)}）
      ${deviationText(metric) ? ' · ' + deviationText(metric) : ''}
    </p>
    <p class="metric__nums">${tags.join(' ')}</p>
  `
  return el
}

/** 渲染右侧面板：总分 + 各分组。 */
export function renderReport(panel: HTMLElement, result: AnalysisResult): void {
  panel.replaceChildren()

  const total = document.createElement('div')
  total.className = 'total'
  const totalColor = scoreColor(result.score)
  total.innerHTML = `
    <div class="total__row">
      <span class="total__num" style="color:${totalColor}">${scoreText(result.score)}</span>
      <span class="total__unit">/ 100</span>
    </div>
    <p class="total__caption">
      面部比例接近人群平均水平的程度，<strong>不是好看程度</strong>。
      单张照片的分数本身有 ±5~8 分的波动，换个角度拍可能明显不同。
    </p>
  `
  panel.append(total)

  for (const area of result.areas) {
    const box = document.createElement('div')
    box.className = 'area is-open'

    const head = document.createElement('button')
    head.type = 'button'
    head.className = 'area__head'
    head.setAttribute('aria-expanded', 'true')
    head.innerHTML = `
      <span class="area__name">${area.label}</span>
      <span class="area__score" style="color:${scoreColor(area.score)}">${scoreText(area.score)}</span>
      <span class="area__chevron" aria-hidden="true">▶</span>
    `

    const body = document.createElement('div')
    body.className = 'area__body'
    for (const metric of area.metrics) body.append(renderMetric(metric))

    head.addEventListener('click', () => {
      const open = box.classList.toggle('is-open')
      body.hidden = !open
      head.setAttribute('aria-expanded', String(open))
    })

    box.append(head, body)
    panel.append(box)
  }
}
