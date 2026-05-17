import { useMemo } from 'react'
import type { CategoryDirectionResult, RunPrompt } from '../../types/run'

interface CategoryConeChartProps {
  directionResult: CategoryDirectionResult
  prompts: RunPrompt[]
  modelId: string
  mode: string
  maximized?: boolean
}

const MODE_COLOR: Record<string, string> = {
  hard:       '#ef4444',
  redirect:   '#f97316',
  disclaimer: '#eab308',
  none:       '#22c55e',
}

const MODE_LABEL: Record<string, string> = {
  hard: 'Hard',
  redirect: 'Redirect',
  disclaimer: 'Disclaimer',
  none: 'None',
}

const W = 240, H = 200
const M = { top: 10, right: 10, bottom: 24, left: 28 }
const PW = W - M.left - M.right
const PH = H - M.top - M.bottom

const toX = (v: number) => M.left + (v + 1) / 2 * PW
const toY = (v: number) => M.top + (1 - (v + 1) / 2) * PH

export const CategoryConeChart = ({ directionResult, prompts, modelId, mode, maximized = false }: CategoryConeChartProps) => {
  const { by_mode } = directionResult

  const peakLayer = useMemo(() => {
    if (!by_mode.hard) return null
    const mag = by_mode.hard.magnitude_per_layer
    return mag.indexOf(Math.max(...mag))
  }, [by_mode.hard])

  const points = useMemo(() => {
    if (!by_mode.hard || peakLayer === null) return []
    return prompts.flatMap(prompt => {
      const result = prompt.model_results[modelId]?.[mode]
      if (!result?.hidden_states_key) return []
      const key = result.hidden_states_key
      const hx = by_mode.hard!.similarity_per_prompt[key]?.[peakLayer]
      if (hx === undefined) return []
      const hy = by_mode.redirect?.similarity_per_prompt[key]?.[peakLayer] ?? 0
      return [{ hx, hy, refusalMode: result.refusal_mode ?? 'none' }]
    })
  }, [by_mode, peakLayer, prompts, modelId, mode])

  if (!by_mode.hard || !by_mode.redirect) return null

  // Cone triangle vertices in pixel space
  const ox = toX(0), oy = toY(0)
  const hardPx = toX(1), hardPy = toY(0)
  const redirPx = toX(0), redirPy = toY(1)

  const ticks = [-1, -0.5, 0, 0.5, 1]

  return (
    <div style={{ marginTop: '8px', borderTop: '1px solid var(--border)', paddingTop: '6px' }}>
      <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '2px' }}>
        direction subspace @ L{peakLayer}
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: maximized ? '360px' : '180px', display: 'block' }}>
        <defs>
          <linearGradient id="cone-face-grad"
            x1={hardPx} y1={hardPy}
            x2={redirPx} y2={redirPy}
            gradientUnits="userSpaceOnUse"
          >
            <stop offset="0%"   stopColor="#ef4444" stopOpacity={0.22} />
            <stop offset="100%" stopColor="#f97316" stopOpacity={0.22} />
          </linearGradient>
          <marker id="arrowhead-hard" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto">
            <polygon points="0 0, 5 2.5, 0 5" fill="#ef4444" opacity={0.7} />
          </marker>
          <marker id="arrowhead-redirect" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto">
            <polygon points="0 0, 5 2.5, 0 5" fill="#f97316" opacity={0.7} />
          </marker>
        </defs>

        {/* Cone face — triangle between origin and the two direction vectors */}
        <polygon
          points={`${ox},${oy} ${hardPx},${hardPy} ${redirPx},${redirPy}`}
          fill="url(#cone-face-grad)"
          stroke="none"
        />

        {/* Axis reference lines */}
        <line x1={M.left} y1={oy} x2={M.left + PW} y2={oy} stroke="var(--border)" strokeWidth={1} />
        <line x1={ox} y1={M.top} x2={ox} y2={M.top + PH} stroke="var(--border)" strokeWidth={1} />

        {/* Plot border */}
        <rect x={M.left} y={M.top} width={PW} height={PH} fill="none" stroke="var(--border)" strokeWidth={0.5} />

        {/* Direction vector arrows */}
        <line
          x1={ox} y1={oy} x2={hardPx - 4} y2={hardPy}
          stroke="#ef4444" strokeWidth={1.5} opacity={0.7}
          markerEnd="url(#arrowhead-hard)"
        />
        <line
          x1={ox} y1={oy} x2={redirPx} y2={redirPy + 4}
          stroke="#f97316" strokeWidth={1.5} opacity={0.7}
          markerEnd="url(#arrowhead-redirect)"
        />

        {/* X axis ticks + labels */}
        {ticks.map(v => (
          <g key={`x${v}`}>
            <line x1={toX(v)} y1={M.top + PH} x2={toX(v)} y2={M.top + PH + 3} stroke="var(--border)" strokeWidth={0.5} />
            <text x={toX(v)} y={M.top + PH + 11} textAnchor="middle" fill="var(--text-muted)" fontSize={7}>
              {v}
            </text>
          </g>
        ))}

        {/* Y axis ticks + labels */}
        {ticks.map(v => (
          <g key={`y${v}`}>
            <line x1={M.left - 3} y1={toY(v)} x2={M.left} y2={toY(v)} stroke="var(--border)" strokeWidth={0.5} />
            <text x={M.left - 5} y={toY(v) + 3} textAnchor="end" fill="var(--text-muted)" fontSize={7}>
              {v}
            </text>
          </g>
        ))}

        {/* Axis labels */}
        <text x={M.left + PW / 2} y={H - 2} textAnchor="middle" fill="var(--text-muted)" fontSize={8}>
          hard sim
        </text>
        <text
          x={9}
          y={M.top + PH / 2}
          textAnchor="middle"
          fill="var(--text-muted)"
          fontSize={8}
          transform={`rotate(-90, 9, ${M.top + PH / 2})`}
        >
          redirect sim
        </text>

        {/* Ray lines from origin to each point */}
        {points.map(({ hx, hy, refusalMode }, idx) => (
          <line
            key={`ray-${idx}`}
            x1={ox} y1={oy}
            x2={toX(hx)} y2={toY(hy)}
            stroke={MODE_COLOR[refusalMode] ?? '#888'}
            strokeWidth={0.75}
            opacity={0.45}
          />
        ))}

        {/* Scatter points */}
        {points.map(({ hx, hy, refusalMode }, idx) => (
          <circle
            key={`pt-${idx}`}
            cx={toX(hx)}
            cy={toY(hy)}
            r={1.5}
            fill={MODE_COLOR[refusalMode] ?? '#888'}
            fillOpacity={0.9}
          />
        ))}
      </svg>

      {/* Legend */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginTop: '2px' }}>
        {(Object.keys(MODE_COLOR) as Array<keyof typeof MODE_COLOR>).map(m => {
          const count = points.filter(p => p.refusalMode === m).length
          if (!count) return null
          return (
            <span key={m} style={{ display: 'flex', alignItems: 'center', gap: '3px', fontSize: '9px', color: 'var(--text-muted)' }}>
              <span style={{ width: '7px', height: '7px', borderRadius: '50%', background: MODE_COLOR[m], display: 'inline-block' }} />
              {MODE_LABEL[m]} ({count})
            </span>
          )
        })}
      </div>
    </div>
  )
}
