import { useEffect, useMemo, useState } from 'react'
import { CATEGORIES, GROUP_LABELS } from '../../types/categories'

interface PcaArrow {
  id: string
  name: string
  peak_layer: number
  magnitude: number
  x: number
  y: number
}

interface ProjectionPayload {
  hard: PcaArrow[]
  redirect: PcaArrow[]
  variance_explained: number[]
}

interface PcaPayload {
  centered:     ProjectionPayload
  uncentered:   ProjectionPayload
  n_categories: number
  hidden_dim:   number
  extraction:   { mode: 'peak' | 'mean'; range?: [number, number] }
}

const W = 480
const H = 460
const MARGIN = { top: 40, right: 40, bottom: 40, left: 40 }
const PLOT_W = W - MARGIN.left - MARGIN.right
const PLOT_H = H - MARGIN.top - MARGIN.bottom

const GROUP_COLOR: Record<string, string> = {
  violence_physical_harm:  '#ef4444',
  weapons_wmd:             '#f97316',
  child_safety:            '#eab308',
  sexual_content:          '#f59e0b',
  cyber_technical_harm:    '#84cc16',
  self_harm_mental_health: '#22c55e',
  hate_discrimination:     '#14b8a6',
  privacy:                 '#06b6d4',
  deception_manipulation:  '#3b82f6',
  misinformation:          '#6366f1',
  illegal_activities:      '#8b5cf6',
  content_policy:          '#a855f7',
  professional_advice:     '#ec4899',
  system_integrity:        '#f43f5e',
  capability_limits:       '#94a3b8',
}

const groupOf = (catId: string): string =>
  CATEGORIES.find(cat => cat.id === catId)?.group ?? 'capability_limits'

const colorOf = (catId: string): string =>
  GROUP_COLOR[groupOf(catId)] ?? '#94a3b8'

interface VectorPlotProps {
  title: string
  subtitle: string
  payload: ProjectionPayload
  markerSuffix: string
}

const VectorPlot = ({ title, subtitle, payload, markerSuffix }: VectorPlotProps) => {
  const layout = useMemo(() => {
    const allArrows = [...payload.hard, ...payload.redirect]
    if (!allArrows.length) return null
    const xs = allArrows.map(arrow => arrow.x)
    const ys = allArrows.map(arrow => arrow.y)
    const xMax = Math.max(Math.abs(Math.max(...xs)), Math.abs(Math.min(...xs))) || 1
    const yMax = Math.max(Math.abs(Math.max(...ys)), Math.abs(Math.min(...ys))) || 1
    const cap = Math.max(xMax, yMax) * 1.1

    const cx = MARGIN.left + PLOT_W / 2
    const cy = MARGIN.top + PLOT_H / 2
    const scale = Math.min(PLOT_W, PLOT_H) / 2 / cap

    return {
      cx,
      cy,
      toX: (value: number) => cx + value * scale,
      toY: (value: number) => cy - value * scale,
    }
  }, [payload])

  if (!layout) return null

  const pc1Percent = (payload.variance_explained[0] * 100).toFixed(1)
  const pc2Percent = (payload.variance_explained[1] * 100).toFixed(1)

  return (
    <div>
      <div style={{ marginBottom: '6px' }}>
        <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text)' }}>
          {title}
        </div>
        <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '2px' }}>
          {subtitle} · PC1 {pc1Percent}% · PC2 {pc2Percent}%
        </div>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
        <defs>
          {Object.entries(GROUP_COLOR).map(([group, color]) => (
            <marker key={group} id={`arrow-${markerSuffix}-${group}`} viewBox="0 0 8 8" refX="6" refY="4" markerWidth="5" markerHeight="5" orient="auto">
              <path d="M0,0 L8,4 L0,8 Z" fill={color} />
            </marker>
          ))}
        </defs>

        <line x1={MARGIN.left} y1={layout.cy} x2={W - MARGIN.right} y2={layout.cy} stroke="var(--border)" strokeWidth={1} />
        <line x1={layout.cx} y1={MARGIN.top}  x2={layout.cx} y2={H - MARGIN.bottom} stroke="var(--border)" strokeWidth={1} />

        <text x={W - MARGIN.right - 4} y={layout.cy - 6} textAnchor="end" fontSize="10" fill="var(--text-muted)">PC1 ({pc1Percent}%)</text>
        <text x={layout.cx + 6}        y={MARGIN.top + 12}                        fontSize="10" fill="var(--text-muted)">PC2 ({pc2Percent}%)</text>

        {payload.redirect.map(arrow => {
          const color = colorOf(arrow.id)
          const x2 = layout.toX(arrow.x)
          const y2 = layout.toY(arrow.y)
          return (
            <line
              key={`redir-${arrow.id}`}
              x1={layout.cx} y1={layout.cy}
              x2={x2}        y2={y2}
              stroke={color}
              strokeWidth={1.5}
              strokeOpacity={0.7}
              strokeDasharray="3 3"
              markerEnd={`url(#arrow-${markerSuffix}-${groupOf(arrow.id)})`}
            />
          )
        })}

        {payload.hard.map(arrow => {
          const color = colorOf(arrow.id)
          const x2 = layout.toX(arrow.x)
          const y2 = layout.toY(arrow.y)
          const tipDx = x2 - layout.cx
          const tipDy = y2 - layout.cy
          const norm = Math.hypot(tipDx, tipDy) || 1
          const labelOffsetX = tipDx / norm * 8
          const labelOffsetY = tipDy / norm * 8

          return (
            <g key={`hard-${arrow.id}`}>
              <line
                x1={layout.cx} y1={layout.cy}
                x2={x2}        y2={y2}
                stroke={color}
                strokeWidth={1.5}
                strokeOpacity={0.9}
                markerEnd={`url(#arrow-${markerSuffix}-${groupOf(arrow.id)})`}
              />
              <text
                x={x2 + labelOffsetX}
                y={y2 + labelOffsetY}
                fontSize="9"
                fill="var(--text)"
                textAnchor={tipDx > 0 ? 'start' : 'end'}
                dominantBaseline={tipDy > 0 ? 'hanging' : 'auto'}
              >
                {arrow.name}
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

export const DirectionVectorChart = () => {
  const [data, setData] = useState<PcaPayload | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetch('/direction_pca.json')
      .then(res => res.ok ? res.json() : Promise.reject(new Error(`status ${res.status}`)))
      .then(setData)
      .catch(err => setError(err.message))
  }, [])

  if (error) return <div style={{ padding: '24px', color: '#ef4444' }}>Failed to load PCA: {error}</div>
  if (!data)  return <div style={{ padding: '24px', color: 'var(--text-muted)' }}>Loading direction embedding…</div>

  const extractionLabel = data.extraction.mode === 'mean' && data.extraction.range
    ? `mean L${data.extraction.range[0]}–L${data.extraction.range[1]}`
    : 'peak layer'

  const presentGroups = Array.from(new Set(
    [...data.centered.hard, ...data.centered.redirect].map(arrow => groupOf(arrow.id))
  ))

  return (
    <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '16px' }}>
      <div style={{ marginBottom: '12px' }}>
        <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text)' }}>
          Per-category ablation directions · {data.n_categories} cats · {data.hidden_dim}-dim → 2D · {extractionLabel}
        </div>
        <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>
          <span style={{ borderBottom: '2px solid currentColor', paddingBottom: '1px' }}>solid = hard</span>
          {'  ·  '}
          <span style={{ borderBottom: '2px dotted currentColor', paddingBottom: '1px' }}>dotted = redirect</span>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
        <VectorPlot
          title="Centered (PCA from centroid)"
          subtitle="arrows = deviations from average refusal direction"
          payload={data.centered}
          markerSuffix="centered"
        />
        <VectorPlot
          title="Uncentered (SVD from origin)"
          subtitle="arrows = absolute directions, cone axis = PC1"
          payload={data.uncentered}
          markerSuffix="uncentered"
        />
      </div>

      <div style={{ marginTop: '12px', display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
        {presentGroups.map(group => (
          <span key={group} style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', fontSize: '10px', color: 'var(--text-muted)' }}>
            <span style={{ width: '10px', height: '2px', background: GROUP_COLOR[group] ?? '#94a3b8' }} />
            {GROUP_LABELS[group] ?? group}
          </span>
        ))}
      </div>
    </div>
  )
}
