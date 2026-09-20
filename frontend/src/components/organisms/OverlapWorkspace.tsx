import { useCallback, useEffect, useMemo, useState } from 'react'
import { listAudits, directionOverlap } from '../../api/ablation'
import type { AuditSummary, DirectionOverlapResponse } from '../../types/ablation'
import { CATEGORIES } from '../../types/categories'

interface OverlapWorkspaceProps {
  run: { run_id: string; sequence?: { model: string; mode: string }[] }
}

const CATEGORY_NAMES: Record<string, string> = {}
for (const cat of CATEGORIES) CATEGORY_NAMES[cat.id] = cat.name
const labelOf = (id: string): string => CATEGORY_NAMES[id] ?? id

// A small palette of primaries. Directions are assigned in order; the overlap
// blend is a straight RGB average of whatever primaries actually share space.
const PRIMARIES: [number, number, number][] = [
  [255, 0, 0],     // red
  [0, 0, 255],     // blue
  [0, 170, 0],     // green
  [255, 165, 0],   // orange
  [160, 32, 240],  // purple
  [0, 200, 200],   // teal
]

const fmtTime = (iso: string | null | undefined): string => {
  if (!iso) return '—'
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString()
}

// --- SVG geometry -----------------------------------------------------------

const W = 520
const H = 480
const MARGIN = { top: 30, right: 30, bottom: 30, left: 30 }

interface Arrow { name: string; x: number; y: number; magnitude?: number }

// Fit the set of arrows into the canvas with a shared scale + center.
const fitLayout = (arrows: Arrow[]) => {
  if (arrows.length === 0) return null
  let maxAbs = 1e-9
  for (const a of arrows) maxAbs = Math.max(maxAbs, Math.abs(a.x), Math.abs(a.y))
  const scale = (Math.min(W, H) / 2 - MARGIN.top - 6) / maxAbs
  const cx = W / 2
  const cy = H / 2
  return {
    toX: (v: number) => cx + v * scale,
    toY: (v: number) => cy - v * scale, // flip y so +y points up like the other charts
    cx,
    cy,
  }
}

// A direction arrow swept into a small cone/sector. Returns an SVG path string.
const sectorPath = (
  tipX: number, tipY: number, dirX: number, dirY: number, length: number, halfAngle: number,
): string => {
  const angle = Math.atan2(dirY, dirX)
  const spread = halfAngle // radians
  const a1 = angle - spread
  const a2 = angle + spread
  const p1x = tipX + length * Math.cos(a1)
  const p1y = tipY + length * Math.sin(a1)
  const p2x = tipX + length * Math.cos(a2)
  const p2y = tipY + length * Math.sin(a2)
  return `M ${tipX} ${tipY} L ${p1x} ${p1y} A ${length} ${length} 0 0 1 ${p2x} ${p2y} Z`
}

// Average the RGB of a set of primaries → the blended fill for an overlap region.
const blend = (colors: [number, number, number][]): string => {
  if (colors.length === 0) return 'rgba(128,128,128,0.30)'
  const r = Math.round(colors.reduce((s, c) => s + c[0], 0) / colors.length)
  const g = Math.round(colors.reduce((s, c) => s + c[1], 0) / colors.length)
  const b = Math.round(colors.reduce((s, c) => s + c[2], 0) / colors.length)
  return `rgba(${r},${g},${b},0.30)`
}

// --- Component --------------------------------------------------------------

export const OverlapWorkspace = ({ run }: OverlapWorkspaceProps) => {
  const [audits, setAudits] = useState<AuditSummary[]>([])
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [data, setData] = useState<DirectionOverlapResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const refreshAudits = useCallback(() => {
    listAudits(run.run_id).then(setAudits).catch(() => setAudits([]))
  }, [run.run_id])

  useEffect(refreshAudits, [refreshAudits])

  const toggle = (path: string) => {
    setSelected(prev => {
      const next = new Set(prev)
      if (next.has(path)) next.delete(path)
      else next.add(path)
      return next
    })
  }

  // Fetch the overlap geometry whenever the selection changes.
  useEffect(() => {
    if (selected.size === 0) { setData(null); return }
    const step = run.sequence?.[0]
    if (!step) { setError('No model/mode in run sequence'); return }
    let cancelled = false
    setLoading(true)
    setError(null)
    directionOverlap({
      run_id: run.run_id,
      model_id: step.model,
      mode: step.mode,
      experiments: Array.from(selected).map(path => ({ path })),
    })
      .then(res => { if (!cancelled) setData(res) })
      .catch(err => { if (!cancelled) setError(err instanceof Error ? err.message : String(err)) })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [selected, run.run_id, run.sequence])

  // Split categories: refused in ANY selected run → Refused section; else Non-refused.
  const { refusedCats, okCats } = useMemo(() => {
    const refused = new Set<string>()
    const ok = new Set<string>()
    for (const exp of data?.experiments ?? []) {
      for (const c of exp.refused_categories) refused.add(c)
      for (const c of exp.ok_categories) ok.add(c)
    }
    // A category is "refused" if it was refused anywhere; drop it from the ok set.
    for (const c of refused) ok.delete(c)
    return { refusedCats: [...refused], okCats: [...ok] }
  }, [data])

  const arrows = data?.arrows ?? {}
  const arrowList = useMemo(() => Object.values(arrows), [arrows])
  const layout = fitLayout(arrowList)

  // Assign each direction a primary color (stable by category id order).
  const colorOf = useCallback((catId: string): [number, number, number] => {
    const all = [...refusedCats, ...okCats]
    const idx = all.indexOf(catId)
    return PRIMARIES[(idx < 0 ? 0 : idx) % PRIMARIES.length]
  }, [refusedCats, okCats])

  // Which directions overlap each other? Two sectors overlap if their tip-to-tip
  // distance is small relative to sector length. We approximate "shared subspace"
  // as: the two arrows point in similar-enough directions AND are close at origin —
  // since all cones share the origin, we blend any pair whose angular separation is
  // under the cone half-angle sum.
  const HALF_ANGLE = 0.35 // rad per side → sectors overlap if angle diff < ~0.7
  const angleOf = (a: Arrow) => Math.atan2(-a.y, a.x) // flip y back to screen space

  const renderSection = (cats: string[], sectionKey: string) => {
    if (!layout || cats.length === 0) return null
    const paths = []
    for (const cat of cats) {
      const arrow = arrows[cat]
      if (!arrow) continue
      const tipX = layout.cx
      const tipY = layout.cy
      const dirX = layout.toX(arrow.x) - tipX
      const dirY = layout.toY(arrow.y) - tipY
      const len = Math.hypot(dirX, dirY) || 1
      // Sector length scaled to the arrow's on-screen length.
      const sectorLen = Math.min(len * 1.4, 200)
      const d = sectorPath(tipX, tipY, dirX / len, dirY / len, sectorLen, HALF_ANGLE)

      // Blend with any other direction (in either section) whose cone overlaps this one.
      const myAngle = angleOf(arrow)
      const overlapping: [number, number, number][] = []
      for (const otherCat of [...refusedCats, ...okCats]) {
        if (otherCat === cat) continue
        const other = arrows[otherCat]
        if (!other) continue
        let diff = Math.abs(angleOf(other) - myAngle)
        if (diff > Math.PI) diff = 2 * Math.PI - diff
        if (diff < HALF_ANGLE * 2) overlapping.push(colorOf(otherCat))
      }
      const fill = blend(overlapping.length ? [colorOf(cat), ...overlapping] : [colorOf(cat)])

      paths.push(
        <g key={sectionKey + '-' + cat}>
          <path d={d} fill={fill} stroke="none" />
          {/* faint center line so the direction is still readable */}
          <line x1={tipX} y1={tipY} x2={tipX + dirX} y2={tipY + dirY}
            stroke={blend([colorOf(cat)])} strokeWidth={1.5} opacity={0.9} />
          <text x={tipX + dirX * 1.08} y={tipY + dirY * 1.08} fontSize={10}
            fill="var(--text-dim)" textAnchor="middle">{labelOf(cat)}</text>
        </g>,
      )
    }
    return paths
  }

  return (
    <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap' }}>
      {/* Left column: experiment list with checkboxes */}
      <div style={{ width: '300px', flexShrink: 0 }}>
        <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '10px' }}>
          Experiments ({selected.size}/{audits.length})
        </div>
        {audits.length === 0 && (
          <div style={{ fontSize: '12px', color: 'var(--text-dim)' }}>
            No audits saved yet. Run an audit first.
          </div>
        )}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
          {audits.map(audit => (
            <label key={audit.path} style={{ display: 'flex', alignItems: 'flex-start', gap: '8px', fontSize: '12px', color: 'var(--text-dim)', cursor: 'pointer' }}>
              <input type="checkbox" checked={selected.has(audit.path)} onChange={() => toggle(audit.path)} />
              <span style={{ flex: 1 }}>
                {fmtTime(audit.created_at)}<br />
                <span style={{ color: 'var(--text-dim)' }}>{audit.n_trials} trials · {audit.n_refused} refused</span>
              </span>
            </label>
          ))}
        </div>
      </div>

      {/* Visual workspace */}
      <div style={{ flex: 1, minWidth: '320px' }}>
        <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '10px' }}>
          Direction-overlap workspace
        </div>
        {error && <div style={{ fontSize: '12px', color: '#ef4444', marginBottom: '10px' }}>{error}</div>}
        {loading && <div style={{ fontSize: '12px', color: 'var(--text-dim)' }}>Computing overlap…</div>}
        {!data && !loading && selected.size > 0 && (
          <div style={{ fontSize: '12px', color: 'var(--text-dim)' }}>Select experiments to project.</div>
        )}

        {data && layout && (
          <>
            <svg viewBox={`0 0 ${W} ${H}`} width={W} height={H}
              style={{ background: 'var(--surface, #111)', borderRadius: '8px', border: '1px solid var(--border, #333)' }}>
              {/* axes */}
              <line x1={MARGIN.left} y1={layout.cy} x2={W - MARGIN.right} y2={layout.cy} stroke="var(--border, #444)" strokeWidth={0.5} />
              <line x1={layout.cx} y1={MARGIN.top} x2={layout.cx} y2={H - MARGIN.bottom} stroke="var(--border, #444)" strokeWidth={0.5} />

              {/* Refused section */}
              {renderSection(refusedCats, 'refused')}
              {/* Non-refused section */}
              {renderSection(okCats, 'ok')}
            </svg>

            <div style={{ display: 'flex', gap: '24px', marginTop: '10px', fontSize: '12px' }}>
              <div>
                <span style={{ color: '#ef4444', fontWeight: 600 }}>Refused ({refusedCats.length})</span>{' '}
                <span style={{ color: 'var(--text-dim)' }}>{refusedCats.map(labelOf).join(', ') || '—'}</span>
              </div>
              <div>
                <span style={{ color: '#22c55e', fontWeight: 600 }}>Non-refused ({okCats.length})</span>{' '}
                <span style={{ color: 'var(--text-dim)' }}>{okCats.map(labelOf).join(', ') || '—'}</span>
              </div>
            </div>
            <p style={{ fontSize: '11px', color: 'var(--text-dim)', marginTop: '8px', lineHeight: 1.5 }}>
              Each cone is a direction's projected PCA arrow swept into a sector. Overlapping cones
              blend their primaries (red + blue → purple); every fill is at alpha 0.30. This is the
              <em>projected</em> overlap in the shared 2D basis, not a full subspace intersection.
            </p>
          </>
        )}
      </div>
    </div>
  )
}
