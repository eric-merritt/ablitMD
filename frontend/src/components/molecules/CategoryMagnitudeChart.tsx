import { useMemo } from 'react'
import { LineChart, Line, XAxis, YAxis, ReferenceLine, Tooltip, ResponsiveContainer } from 'recharts'
import type { CategoryDirectionResult } from '../../types/run'

interface CategoryMagnitudeChartProps {
  directionResult: CategoryDirectionResult
  maximized?: boolean
}

const HARD_COLOR     = '#ef4444'
const REDIRECT_COLOR = '#f97316'
const ALIGN_COLOR    = 'var(--accent)'

const peakIndexOf = (values: number[] | undefined): number | null =>
  values?.length ? values.indexOf(Math.max(...values)) : null

export const CategoryMagnitudeChart = ({ directionResult, maximized = false }: CategoryMagnitudeChartProps) => {
  const hardMag     = directionResult.by_mode.hard?.magnitude_per_layer
  const redirectMag = directionResult.by_mode.redirect?.magnitude_per_layer
  const alignment   = directionResult.alignment['hard_vs_redirect']

  const nLayers = Math.max(hardMag?.length ?? 0, redirectMag?.length ?? 0)

  const chartData = useMemo(() => {
    return Array.from({ length: nLayers }, (_, L) => ({
      layer: L,
      hard:     hardMag?.[L],
      redirect: redirectMag?.[L],
      align:    alignment?.[L],
    }))
  }, [nLayers, hardMag, redirectMag, alignment])

  const hardPeak     = useMemo(() => peakIndexOf(hardMag),     [hardMag])
  const redirectPeak = useMemo(() => peakIndexOf(redirectMag), [redirectMag])

  if (!hardMag && !redirectMag) return null

  const yMax = Math.max(
    ...(hardMag ?? [0]),
    ...(redirectMag ?? [0]),
    1
  )

  return (
    <div style={{ marginTop: '8px', borderTop: '1px solid var(--border)', paddingTop: '6px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: 'var(--text-muted)', marginBottom: '2px' }}>
        <span>direction magnitude per layer</span>
        <span>
          {hardPeak     !== null && <span style={{ color: HARD_COLOR }}>     hard L{hardPeak} </span>}
          {redirectPeak !== null && <span style={{ color: REDIRECT_COLOR }}> redir L{redirectPeak}</span>}
        </span>
      </div>

      <ResponsiveContainer width="100%" height={maximized ? 200 : 110}>
        <LineChart data={chartData} margin={{ top: 4, right: 4, bottom: 4, left: -20 }}>
          <XAxis dataKey="layer" tick={{ fontSize: 9, fill: 'var(--text-muted)' }} />
          <YAxis yAxisId="mag"   domain={[0, yMax]}    tick={{ fontSize: 9, fill: 'var(--text-muted)' }} />
          <YAxis yAxisId="align" orientation="right" domain={[-1, 1]} tick={{ fontSize: 9, fill: 'var(--text-muted)' }} hide={!maximized} />

          <ReferenceLine y={0}   yAxisId="mag"   stroke="var(--border)" />
          {hardPeak     !== null && <ReferenceLine x={hardPeak}     stroke={`${HARD_COLOR}66`}     strokeDasharray="2 2" yAxisId="mag" />}
          {redirectPeak !== null && <ReferenceLine x={redirectPeak} stroke={`${REDIRECT_COLOR}66`} strokeDasharray="2 2" yAxisId="mag" />}

          <Tooltip
            contentStyle={{ background: 'var(--surface-2)', border: '1px solid var(--border)', borderRadius: '4px', fontSize: '10px' }}
            labelStyle={{ color: 'var(--text-muted)' }}
            formatter={(val, name) => [val != null ? (+val).toFixed(3) : '—', String(name)]}
          />

          {hardMag     && <Line yAxisId="mag"   type="monotone" dataKey="hard"     stroke={HARD_COLOR}     strokeWidth={1.5} dot={false} connectNulls />}
          {redirectMag && <Line yAxisId="mag"   type="monotone" dataKey="redirect" stroke={REDIRECT_COLOR} strokeWidth={1.5} dot={false} connectNulls />}
          {alignment   && <Line yAxisId="align" type="monotone" dataKey="align"    stroke={ALIGN_COLOR}    strokeWidth={1}   strokeDasharray="3 2" dot={false} connectNulls />}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
