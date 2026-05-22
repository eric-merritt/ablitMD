import { useMemo, useState } from 'react'
import { LineChart, Line, XAxis, YAxis, ReferenceLine, Tooltip, ResponsiveContainer } from 'recharts'
import { CATEGORIES } from '../../types/categories'
import { CategoryConeChart } from './CategoryConeChart'
import type { CategoryDirectionResult, RunPrompt, RefusalMode } from '../../types/run'

interface CategoryDirectionChartProps {
  categoryName: string
  categoryId: string
  directionResult: CategoryDirectionResult
  prompts: RunPrompt[]
  modelId: string
  mode: string
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

const ExpandIcon = () => (
  <svg width="11" height="11" viewBox="0 0 11 11" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="7,1 10,1 10,4" />
    <polyline points="4,10 1,10 1,7" />
    <line x1="10" y1="1" x2="6" y2="5" />
    <line x1="1" y1="10" x2="5" y2="6" />
  </svg>
)

const CompressIcon = () => (
  <svg width="11" height="11" viewBox="0 0 11 11" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="10,4 7,4 7,1" />
    <polyline points="1,7 4,7 4,10" />
    <line x1="7" y1="4" x2="11" y2="0" />
    <line x1="4" y1="7" x2="0" y2="11" />
  </svg>
)

const meanOf = (vals: (number | undefined)[]) => {
  const valid = vals.filter((v): v is number => v !== undefined)
  return valid.length ? valid.reduce((a, b) => a + b, 0) / valid.length : undefined
}

export const CategoryDirectionChart = ({
  categoryName,
  categoryId,
  directionResult,
  prompts,
  modelId,
  mode,
}: CategoryDirectionChartProps) => {
  const [maximized, setMaximized] = useState(false)
  // reference direction for the chart: prefer hard, else fall back to whatever exists
  const refMode = directionResult.by_mode.hard
    ? 'hard'
    : directionResult.by_mode.redirect
      ? 'redirect'
      : directionResult.by_mode.disclaimer
        ? 'disclaimer'
        : null
  const refEntry = refMode ? directionResult.by_mode[refMode] : undefined

  const modeGroups = useMemo(() => {
    const groups: Record<string, string[]> = { hard: [], redirect: [], disclaimer: [], none: [] }
    for (const prompt of prompts) {
      if (prompt.category !== categoryId) continue
      const result = prompt.model_results[modelId]?.[mode]
      if (result?.hidden_states_key) {
        const m = result.refusal_mode ?? 'none'
        groups[m]?.push(result.hidden_states_key)
      }
    }
    return groups
  }, [prompts, categoryId, modelId, mode])

  const chartData = useMemo(() => {
    if (!refEntry) return []
    const { similarity_per_prompt } = refEntry
    const nLayers = refEntry.magnitude_per_layer.length

    return Array.from({ length: nLayers }, (_, L) => {
      const point: Record<string, number | undefined> = { layer: L }
      for (const m of ['hard', 'redirect', 'disclaimer', 'none'] as RefusalMode[]) {
        const keys = modeGroups[m] ?? []
        if (keys.length) {
          const val = meanOf(keys.map(k => similarity_per_prompt[k]?.[L]))
          point[m] = (m === 'none' || m === 'disclaimer') && val !== undefined ? -val : val
        }
      }
      return point
    })
  }, [refEntry, modeGroups])

  const peakLayer = useMemo(() => {
    if (!refEntry) return null
    const mag = refEntry.magnitude_per_layer
    return mag.indexOf(Math.max(...mag))
  }, [refEntry])

  const alignmentLabel = useMemo(() => {
    const a = directionResult.alignment['hard_vs_redirect']
    if (!a?.length) return null
    const mean = a.reduce((s, v) => s + v, 0) / a.length
    return `${Math.round(mean * 100)}%`
  }, [directionResult.alignment])

  const triggerOverlap = useMemo(() => {
    const { trigger_meta } = directionResult
    if (!trigger_meta || !refEntry || peakLayer === null) return []
    const { trigger_similarity_per_prompt } = refEntry
    if (!trigger_similarity_per_prompt) return []

    const bySource: Record<string, number[]> = {}
    for (const [key, meta] of Object.entries(trigger_meta)) {
      const sims = trigger_similarity_per_prompt[key]
      if (!sims) continue
      const val = sims[peakLayer]
      if (val === undefined) continue
      bySource[meta.source_category] ??= []
      bySource[meta.source_category].push(val)
    }

    return Object.entries(bySource).map(([catId, vals]) => {
      const mean = vals.reduce((a, b) => a + b, 0) / vals.length
      const name = CATEGORIES.find(c => c.id === catId)?.name ?? catId
      return { catId, name, mean }
    }).sort((a, b) => b.mean - a.mean)
  }, [directionResult, peakLayer])

  if (!refEntry) {
    return (
      <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '12px' }}>
        <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-dim)', marginBottom: '4px' }}>{categoryName}</div>
        <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>No direction data</div>
      </div>
    )
  }

  const presentModes = (['hard', 'redirect', 'disclaimer', 'none'] as const).filter(
    m => (modeGroups[m]?.length ?? 0) > 0
  )

  return (
    <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '12px', gridColumn: maximized ? '1 / -1' : undefined }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '6px' }}>
        <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-dim)' }}>{categoryName}</div>
        <button
          onClick={() => setMaximized(m => !m)}
          style={{ background: 'none', border: 'none', padding: '2px', color: 'var(--text-muted)', cursor: 'pointer', lineHeight: 0, flexShrink: 0 }}
        >
          {maximized ? <CompressIcon /> : <ExpandIcon />}
        </button>
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', marginBottom: '8px' }}>
        {presentModes.map(m => (
          <span key={m} style={{ fontSize: '10px', padding: '1px 6px', borderRadius: '3px', background: `${MODE_COLOR[m]}22`, color: MODE_COLOR[m], border: `1px solid ${MODE_COLOR[m]}44` }}>
            {MODE_LABEL[m]} {directionResult.by_mode[m as 'hard' | 'redirect' | 'disclaimer']?.sample_count ?? modeGroups[m]?.length}
          </span>
        ))}
        {peakLayer !== null && (
          <span style={{ fontSize: '10px', padding: '1px 6px', borderRadius: '3px', background: 'var(--surface-3)', color: 'var(--text-muted)', border: '1px solid var(--border)' }}>
            peak L{peakLayer}
          </span>
        )}
        {alignmentLabel && (
          <span style={{ fontSize: '10px', padding: '1px 6px', borderRadius: '3px', background: 'var(--surface-3)', color: 'var(--text-muted)', border: '1px solid var(--border)' }}>
            H∥R {alignmentLabel}
          </span>
        )}
      </div>

      <ResponsiveContainer width="100%" height={maximized ? 300 : 150}>
        <LineChart data={chartData} margin={{ top: 4, right: 4, bottom: 4, left: -20 }}>
          <XAxis dataKey="layer" tick={{ fontSize: 9, fill: 'var(--text-muted)' }} />
          <YAxis domain={[-1, 1]} tick={{ fontSize: 9, fill: 'var(--text-muted)' }} />
          <ReferenceLine y={0} stroke="var(--border)" />
          {peakLayer !== null && (
            <ReferenceLine x={peakLayer} stroke="var(--border-2)" strokeDasharray="3 3" />
          )}
          <Tooltip
            contentStyle={{ background: 'var(--surface-2)', border: '1px solid var(--border)', borderRadius: '4px', fontSize: '10px' }}
            labelStyle={{ color: 'var(--text-muted)' }}
            formatter={(val, name) => [val != null ? (+val).toFixed(3) : '—', MODE_LABEL[String(name)] ?? String(name)]}
          />
          {presentModes.map(m => (
            <Line
              key={m}
              type="monotone"
              dataKey={m}
              stroke={MODE_COLOR[m]}
              strokeWidth={m === 'none' ? 1 : 1.5}
              strokeDasharray={m === 'none' ? '4 2' : undefined}
              dot={false}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>

      <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '4px' }}>
        mean similarity vs {refMode} direction
      </div>

      {triggerOverlap.length > 0 && (
        <div style={{ marginTop: '8px', borderTop: '1px solid var(--border)', paddingTop: '6px' }}>
          <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '4px' }}>
            trigger overlap @ L{peakLayer}
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '3px' }}>
            {triggerOverlap.map(({ catId, name, mean }) => {
              const intensity = Math.abs(mean)
              const bg = mean > 0
                ? `rgba(91,156,246,${(intensity * 0.35).toFixed(2)})`
                : `rgba(239,68,68,${(intensity * 0.35).toFixed(2)})`
              const color = mean > 0 ? 'var(--accent)' : '#ef4444'
              return (
                <span
                  key={catId}
                  title={`${name}: ${mean.toFixed(3)}`}
                  style={{ fontSize: '10px', padding: '1px 6px', borderRadius: '3px', background: bg, color, border: `1px solid ${color}44`, cursor: 'default' }}
                >
                  {name} {mean > 0 ? '+' : ''}{mean.toFixed(2)}
                </span>
              )
            })}
          </div>
        </div>
      )}

      <CategoryConeChart
        directionResult={directionResult}
        prompts={prompts}
        modelId={modelId}
        mode={mode}
        maximized={maximized}
      />
    </div>
  )
}
