import type { DirectionOverlapResponse } from '../../types/ablation'
import { CATEGORIES } from '../../types/categories'

const CATEGORY_NAMES: Record<string, string> = {}
for (const cat of CATEGORIES) CATEGORY_NAMES[cat.id] = cat.name
const labelOf = (id: string): string => CATEGORY_NAMES[id] ?? id

interface OverlapStatsProps {
  data: DirectionOverlapResponse
}

export const OverlapStats = ({ data }: OverlapStatsProps) => {
  const arrows = data.arrows
  const cats = Object.keys(arrows)
  const totalCats = cats.length || 1

  // Aggregate refused/ok across all selected experiments.
  const refusedSet = new Set<string>()
  const okSet = new Set<string>()
  for (const exp of data.experiments) {
    for (const c of exp.refused_categories) refusedSet.add(c)
    for (const c of exp.ok_categories) okSet.add(c)
  }
  for (const c of refusedSet) okSet.delete(c)

  const nRefused = refusedSet.size
  const nOk = okSet.size
  const pctRefused = Math.round((nRefused / totalCats) * 100)
  const pctOk = Math.round((nOk / totalCats) * 100)

  // Direction alignment: for each pair, cosine of their 2D projection.
  const pairs: { a: string; b: string; cos: number }[] = []
  for (let i = 0; i < cats.length; i++) {
    for (let j = i + 1; j < cats.length; j++) {
      const a = arrows[cats[i]]
      const b = arrows[cats[j]]
      const dot = a.x * b.x + a.y * b.y
      const magA = Math.hypot(a.x, a.y) || 1
      const magB = Math.hypot(b.x, b.y) || 1
      pairs.push({ a: cats[i], b: cats[j], cos: dot / (magA * magB) })
    }
  }
  pairs.sort((p, q) => q.cos - p.cos)

  // Categories that share direction space with the most others (cos > 0.85).
  const alignedCount: Record<string, number> = {}
  for (const cat of cats) alignedCount[cat] = 0
  for (const p of pairs) {
    if (p.cos > 0.85) {
      alignedCount[p.a]++
      alignedCount[p.b]++
    }
  }
  const mostShared = Object.entries(alignedCount)
    .filter(([, n]) => n > 0)
    .sort((a, b) => b[1] - a[1])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '12px' }}>
      {/* Refusal rate */}
      <div>
        <span style={{ color: 'var(--danger)', fontWeight: 600 }}>{pctRefused}%</span>
        <span style={{ color: 'var(--text-dim)' }}> refused ({nRefused}/{totalCats})</span>
      </div>
      <div>
        <span style={{ color: 'var(--ok)', fontWeight: 600 }}>{pctOk}%</span>
        <span style={{ color: 'var(--text-dim)' }}> compliant ({nOk}/{totalCats})</span>
      </div>

      {/* Top aligned direction pairs */}
      {pairs.length > 0 && (
        <div style={{ marginTop: '8px', display: 'flex', flexDirection: 'column', gap: '3px' }}>
          <span style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Shared directions</span>
          {pairs.slice(0, 4).map((p, i) => (
            <span key={i} style={{ color: p.cos > 0.9 ? 'var(--ok)' : 'var(--text-dim)', fontSize: '11px' }}>
              {labelOf(p.a)} ↔ {labelOf(p.b)} · {(p.cos * 100).toFixed(0)}%
            </span>
          ))}
        </div>
      )}

      {/* Most-shared categories */}
      {mostShared.length > 0 && (
        <div style={{ marginTop: '4px', display: 'flex', flexDirection: 'column', gap: '3px' }}>
          <span style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Most shared</span>
          {mostShared.slice(0, 3).map(([cat, n]) => (
            <span key={cat} style={{ color: 'var(--text-dim)', fontSize: '11px' }}>
              {labelOf(cat)} ({n})
            </span>
          ))}
        </div>
      )}

      {/* Per-experiment breakdown */}
      {data.experiments.length > 0 && (
        <div style={{ marginTop: '8px', display: 'flex', flexDirection: 'column', gap: '3px' }}>
          <span style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Experiments</span>
          {data.experiments.map(exp => (
            <span key={exp.path} style={{ color: 'var(--text-dim)', fontSize: '11px' }}>
              {exp.refused_categories.length}/{totalCats} refused · {new Date(exp.created_at).toLocaleDateString()}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}
