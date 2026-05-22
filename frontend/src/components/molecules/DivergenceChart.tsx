import { useMemo } from 'react'
import { LineChart, Line, XAxis, YAxis, ReferenceLine, Tooltip, ResponsiveContainer } from 'recharts'
import type { ModeDivergence } from '../../types/ablation'

interface DivergenceChartProps {
  hard?: ModeDivergence
  redirect?: ModeDivergence
  onset: number
  split: number
  lastLayer?: number
  factorA?: number
  factorB?: number
}

const HARD_COLOR     = '#ef4444'
const REDIRECT_COLOR = '#f97316'
const MAG_COLOR      = 'var(--text-muted)'

export const DivergenceChart = ({ hard, redirect, onset, split, lastLayer, factorA, factorB }: DivergenceChartProps) => {
  const nLayers = Math.max(hard?.clumping.length ?? 0, redirect?.clumping.length ?? 0)

  const chartData = useMemo(() => {
    const last = lastLayer ?? nLayers - 1
    const fA = factorA ?? 0
    const fB = factorB ?? 0
    const magMax = Math.max(...(hard?.magnitude ?? [1]), 1)
    return Array.from({ length: nLayers }, (_, layer) => {
      const mag = hard?.magnitude[layer]
      let postAblation: number | undefined
      if (mag != null) {
        const factor = layer >= onset && layer <= split ? fA
          : layer > split && layer <= last ? fB
          : 0
        postAblation = (mag * Math.abs(1 - factor)) / magMax
      }
      return {
        layer,
        hardClump:     hard?.clumping[layer],
        redirectClump: redirect?.clumping[layer],
        magnitude:     mag,
        postAblation,
      }
    })
  }, [nLayers, hard, redirect, onset, split, lastLayer, factorA, factorB])

  if (nLayers === 0) return null

  const magMax = Math.max(...(hard?.magnitude ?? [0]), 1)

  return (
    <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '16px' }}>
      <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text)', marginBottom: '2px' }}>
        Cross-category clumping per layer
      </div>
      <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '8px' }}>
        clumping near 1 = categories share a direction · the drop marks divergence
      </div>

      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={chartData} margin={{ top: 8, right: 8, bottom: 4, left: -20 }}>
          <XAxis dataKey="layer" tick={{ fontSize: 9, fill: 'var(--text-muted)' }} />
          <YAxis yAxisId="clump" domain={[0, 1]} tick={{ fontSize: 9, fill: 'var(--text-muted)' }} />
          <YAxis yAxisId="mag" orientation="right" domain={[0, magMax]} tick={{ fontSize: 9, fill: 'var(--text-muted)' }} />

          <ReferenceLine x={onset} yAxisId="clump" stroke="var(--accent)" strokeDasharray="3 2"
            label={{ value: 'onset', fontSize: 9, fill: 'var(--accent)', position: 'top' }} />
          <ReferenceLine x={split} yAxisId="clump" stroke="var(--accent)"
            label={{ value: 'split', fontSize: 9, fill: 'var(--accent)', position: 'top' }} />

          <Tooltip
            contentStyle={{ background: 'var(--surface-2)', border: '1px solid var(--border)', borderRadius: '4px', fontSize: '10px' }}
            labelStyle={{ color: 'var(--text-muted)' }}
            formatter={(val, name) => [val != null ? (+val).toFixed(3) : '—', String(name)]}
          />

          <Line yAxisId="mag" type="monotone" dataKey="magnitude" name="magnitude"
            stroke={MAG_COLOR} strokeWidth={1} strokeDasharray="2 3" dot={false} connectNulls />
          { (factorA != null || factorB != null) && (
            <Line yAxisId="clump" type="monotone" dataKey="postAblation" name="post-ablation"
              stroke="#60a5fa" strokeWidth={1.5} strokeDasharray="3 2" dot={false} connectNulls />
          ) }
          {hard && <Line yAxisId="clump" type="monotone" dataKey="hardClump" name="hard"
            stroke={HARD_COLOR} strokeWidth={1.8} dot={false} connectNulls />}
          {redirect && <Line yAxisId="clump" type="monotone" dataKey="redirectClump" name="redirect"
            stroke={REDIRECT_COLOR} strokeWidth={1.5} strokeDasharray="4 2" dot={false} connectNulls />}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
