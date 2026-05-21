import type { DirectionCompareRow } from '../../types/ablation'

interface DirectionCompareChartProps {
  rows: DirectionCompareRow[]
}

const cosineColor = (value: number) => {
  const abs = Math.abs(value)
  if (abs >= 0.85) return '#6ee7b7'  // very aligned — green
  if (abs >= 0.6)  return '#fde68a'  // moderate — yellow
  return '#fca5a5'                    // divergent — red
}

const MagnitudeBar = ({ value, color, max }: { value: number | null; color: string; max: number }) => (
  <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: '6px', minWidth: 0 }}>
    <div style={{ flex: 1, height: '6px', background: 'var(--surface-3)', borderRadius: '3px', overflow: 'hidden' }}>
      { value !== null && (
        <div style={{
          height: '100%', borderRadius: '3px', background: color,
          width: `${ Math.min((value / max) * 100, 100) }%`,
        }} />
      ) }
    </div>
    <span style={{ fontSize: '10px', color: 'var(--text-muted)', width: '36px', textAlign: 'right', flexShrink: 0 }}>
      { value !== null ? value.toFixed(3) : '—' }
    </span>
  </div>
)

const LayerRow = ({ row, maxMag }: { row: DirectionCompareRow; maxMag: number }) => (
  <div style={{
    display: 'grid', gridTemplateColumns: '36px 1fr 1fr 52px',
    gap: '8px', alignItems: 'center', padding: '3px 0',
    borderTop: '1px solid var(--border)',
  }}>
    <span style={{ fontSize: '10px', color: 'var(--text-muted)', textAlign: 'right' }}>{ row.layer }</span>
    <MagnitudeBar value={ row.ablitmd_magnitude } color="var(--accent)" max={ maxMag } />
    <MagnitudeBar value={ row.classic_magnitude } color="#a78bfa" max={ maxMag } />
    <span style={{
      fontSize: '10px', fontWeight: 600, textAlign: 'center',
      color: row.cosine_similarity !== null ? cosineColor(row.cosine_similarity) : 'var(--text-muted)',
    }}>
      { row.cosine_similarity !== null ? row.cosine_similarity.toFixed(3) : '—' }
    </span>
  </div>
)

export const DirectionCompareChart = ({ rows }: DirectionCompareChartProps) => {
  if (rows.length === 0) return null

  const maxMag = Math.max(
    ...rows.flatMap(row => [row.ablitmd_magnitude ?? 0, row.classic_magnitude ?? 0])
  ) || 1

  return (
    <div style={{
      background: 'var(--surface)', border: '1px solid var(--border)',
      borderRadius: 'var(--radius)', padding: '12px',
    }}>
      <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text)', marginBottom: '8px' }}>
        Direction comparison · per layer
      </div>
      <div style={{
        display: 'grid', gridTemplateColumns: '36px 1fr 1fr 52px',
        gap: '8px', padding: '0 0 6px 0',
      }}>
        <span />
        <span style={{ fontSize: '10px', color: 'var(--accent)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>ablitMD magnitude</span>
        <span style={{ fontSize: '10px', color: '#a78bfa', textTransform: 'uppercase', letterSpacing: '0.05em' }}>classic magnitude</span>
        <span style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', textAlign: 'center' }}>cosine</span>
      </div>
      <div style={{ maxHeight: '320px', overflow: 'auto' }}>
        { rows.map(row => <LayerRow key={ row.layer } row={ row } maxMag={ maxMag } /> ) }
      </div>
    </div>
  )
}
