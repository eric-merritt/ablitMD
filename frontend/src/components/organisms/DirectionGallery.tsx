import { useEffect, useMemo, useState } from 'react'
import { CATEGORIES, GROUP_COLORS } from '../../types/categories'
import { inferenceDirectionPca, type PcaPayload, type PcaArrow } from '../../api/inference'
import type { Run } from '../../types/run'

// ---------------------------------------------------------------------------
// Shared data shape: one row per category with its 2D projection + magnitude.
// Every representation below renders from this same list, so switching the
// radio only changes the geometry, never the underlying vectors.
// ---------------------------------------------------------------------------
interface Vec {
  id: string
  name: string
  group: string
  x: number
  y: number
  magnitude: number
  peak_layer: number
  kind: 'hard' | 'redirect'
}

const colorOf = (group: string): string => GROUP_COLORS[group] ?? '#94a3b8'
const groupOf = (id: string): string => CATEGORIES.find(c => c.id === id)?.group ?? 'capability_limits'

type ViewId = 'arrows' | 'parallel' | 'radar' | 'heatmap' | 'scatter'

interface ViewDef {
  id: ViewId
  label: string
  hint: string
}

const VIEWS: ViewDef[] = [
  { id: 'arrows',    label: 'Arrows (from origin)',        hint: 'classic — line + head per vector, length = magnitude' },
  { id: 'parallel',  label: 'Parallel coordinates',         hint: 'one polyline per category across PC1/PC2/magnitude/layer' },
  { id: 'radar',     label: 'Radar / spider',               hint: 'magnitude on a spoke per category, grouped ring' },
  { id: 'heatmap',   label: 'Heatmap',                      hint: 'rows = category, cols = PC1/PC2/mag/layer, color = value' },
  { id: 'scatter',   label: 'Dot scatter (no arrows)',      hint: 'just the projected points, colored by group' },
]

// ===========================================================================
// 1. ARROWS — same geometry as DirectionVectorChart, self-contained here.
// ===========================================================================
const ArrowsView = ({ vecs }: { vecs: Vec[] }) => {
  const W = 480, H = 460, M = 40
  const PW = W - 2 * M, PH = H - 2 * M
  const cx = M + PW / 2, cy = M + PH / 2
  const cap = Math.max(...vecs.map(v => Math.max(Math.abs(v.x), Math.abs(v.y)))) * 1.1 || 1
  const scale = Math.min(PW, PH) / 2 / cap
  const tx = (v: number) => cx + v * scale
  const ty = (v: number) => cy - v * scale

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={M} y1={cy} x2={W - M} y2={cy} stroke="var(--border)" strokeWidth={1} />
      <line x1={cx} y1={M} x2={cx} y2={H - M} stroke="var(--border)" strokeWidth={1} />
      {vecs.map(v => {
        const c = colorOf(v.group)
        const x2 = tx(v.x), y2 = ty(v.y)
        const dash = v.kind === 'redirect' ? '3 3' : undefined
        return (
          <g key={v.id}>
            <line x1={cx} y1={cy} x2={x2} y2={y2} stroke={c} strokeWidth={1.5} strokeOpacity={0.85} strokeDasharray={dash} />
            <circle cx={x2} cy={y2} r={2.5} fill={c} />
          </g>
        )
      })}
    </svg>
  )
}

// ===========================================================================
// 2. PARALLEL COORDINATES — one vertical axis per feature, polyline per cat.
// ===========================================================================
const PC_FEATURES = ['PC1', 'PC2', 'magnitude', 'peak_layer'] as const

const ParallelView = ({ vecs }: { vecs: Vec[] }) => {
  const W = 720, H = 360, M = { t: 24, r: 24, b: 28, l: 24 }
  const PW = W - M.l - M.r, PH = H - M.t - M.b
  const n = PC_FEATURES.length
  const xAt = (i: number) => M.l + (PW / (n - 1)) * i

  // per-feature min/max for scaling each axis independently
  const ranges = useMemo(() => {
    return PC_FEATURES.map((f, i) => {
      const vals = vecs.map(v => featureValue(v, i))
      let lo = Math.min(...vals), hi = Math.max(...vals)
      if (hi - lo < 1e-9) { lo -= 1; hi += 1 }
      return { lo, hi }
    })
  }, [vecs])

  const yAt = (i: number, val: number) => M.t + PH - ((val - ranges[i].lo) / (ranges[i].hi - ranges[i].lo)) * PH

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
      {PC_FEATURES.map((f, i) => (
        <g key={f}>
          <line x1={xAt(i)} y1={M.t} x2={xAt(i)} y2={H - M.b} stroke="var(--border)" strokeWidth={1} />
          <text x={xAt(i)} y={H - M.b + 16} textAnchor="middle" fontSize="10" fill="var(--text-muted)">{f}</text>
        </g>
      ))}
      {vecs.map(v => {
        const c = colorOf(v.group)
        const pts = PC_FEATURES.map((_, i) => `${xAt(i)},${yAt(i, featureValue(v, i)).toFixed(1)}`).join(' ')
        return <polyline key={v.id} points={pts} fill="none" stroke={c} strokeWidth={1.2} strokeOpacity={0.55} />
      })}
    </svg>
  )
}

const featureValue = (v: Vec, i: number): number => {
  switch (i) {
    case 0: return v.x
    case 1: return v.y
    case 2: return v.magnitude
    default: return v.peak_layer
  }
}

// ===========================================================================
// 3. RADAR / SPIDER — one spoke per category, radius = magnitude.
// ===========================================================================
const RadarView = ({ vecs }: { vecs: Vec[] }) => {
  const W = 480, H = 460, cx = W / 2, cy = H / 2
  const R = Math.min(W, H) / 2 - 30
  const maxMag = Math.max(...vecs.map(v => v.magnitude)) || 1
  const n = vecs.length

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
      {[0.25, 0.5, 0.75, 1].map(f => (
        <circle key={f} cx={cx} cy={cy} r={R * f} fill="none" stroke="var(--border)" strokeWidth={1} />
      ))}
      {vecs.map((v, i) => {
        const ang = (i / n) * 2 * Math.PI - Math.PI / 2
        const r = (v.magnitude / maxMag) * R
        const x = cx + Math.cos(ang) * r
        const y = cy + Math.sin(ang) * r
        const lx = cx + Math.cos(ang) * (R + 12)
        const ly = cy + Math.sin(ang) * (R + 12)
        return (
          <g key={v.id}>
            <line x1={cx} y1={cy} x2={x} y2={y} stroke={colorOf(v.group)} strokeWidth={1.5} strokeOpacity={0.8} />
            <circle cx={x} cy={y} r={3} fill={colorOf(v.group)} />
            <text x={lx} y={ly} textAnchor="middle" dominantBaseline="middle" fontSize="7" fill="var(--text-dim)">{v.id}</text>
          </g>
        )
      })}
    </svg>
  )
}

// ===========================================================================
// 4. HEATMAP — rows = category, cols = features, cell color = normalized val.
// ===========================================================================
const HeatmapView = ({ vecs }: { vecs: Vec[] }) => {
  const rowH = 16, labelW = 150, topH = 22
  const W = 720, H = topH + vecs.length * rowH + 8
  const colW = (W - labelW - 8) / PC_FEATURES.length

  // normalize each column to [0,1] for color
  const norm = useMemo(() => {
    return PC_FEATURES.map((_, i) => {
      const vals = vecs.map(v => featureValue(v, i))
      const lo = Math.min(...vals), hi = Math.max(...vals)
      return (v: Vec) => (hi - lo < 1e-9 ? 0.5 : (featureValue(v, i) - lo) / (hi - lo))
    })
  }, [vecs])

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
      {PC_FEATURES.map((f, i) => (
        <text key={f} x={labelW + i * colW + colW / 2} y={14} textAnchor="middle" fontSize="10" fill="var(--text-muted)">{f}</text>
      ))}
      {vecs.map((v, r) => (
        <g key={v.id}>
          <text x={4} y={topH + r * rowH + rowH / 2 + 3} fontSize="9" fill="var(--text-dim)">{v.id}</text>
          {PC_FEATURES.map((_, i) => {
            const t = norm[i](v)
            // blue (low) -> red (high) through neutral
            const fill = `hsl(${240 - t * 240}, 75%, ${30 + t * 25}%)`
            return <rect key={i} x={labelW + i * colW + 1} y={topH + r * rowH + 1} width={colW - 2} height={rowH - 2} fill={fill} rx={2} />
          })}
        </g>
      ))}
    </svg>
  )
}

// ===========================================================================
// 5. DOT SCATTER — projected points only, no arrows from origin.
// ===========================================================================
const ScatterView = ({ vecs }: { vecs: Vec[] }) => {
  const W = 480, H = 460, M = 40
  const PW = W - 2 * M, PH = H - 2 * M
  const cx = M + PW / 2, cy = M + PH / 2
  const cap = Math.max(...vecs.map(v => Math.max(Math.abs(v.x), Math.abs(v.y)))) * 1.1 || 1
  const scale = Math.min(PW, PH) / 2 / cap

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
      <line x1={M} y1={cy} x2={W - M} y2={cy} stroke="var(--border)" strokeWidth={1} />
      <line x1={cx} y1={M} x2={cx} y2={H - M} stroke="var(--border)" strokeWidth={1} />
      {vecs.map(v => (
        <circle key={v.id} cx={cx + v.x * scale} cy={cy - v.y * scale} r={4} fill={colorOf(v.group)} fillOpacity={0.85} stroke="var(--surface)" strokeWidth={1} />
      ))}
    </svg>
  )
}

// ===========================================================================
// Container: fetch PCA once, radio list to pick the representation.
// ===========================================================================
export const DirectionGallery = ({ run }: { run: Run }) => {
  const [data, setData] = useState<PcaPayload | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [view, setView] = useState<ViewId>('arrows')

  const step = run.sequence[0]

  useEffect(() => {
    if (!step) { setError('run has no sequence step'); return }
    setData(null); setError(null)
    inferenceDirectionPca({ run_id: run.run_id, model_id: step.model, mode: step.mode })
      .then(setData)
      .catch(err => setError(err.message))
  }, [run.run_id, step?.model, step?.mode])

  const vecs: Vec[] = useMemo(() => {
    if (!data) return []
    const out: Vec[] = []
    for (const kind of ['hard', 'redirect'] as const) {
      for (const a of data.centered[kind]) {
        out.push({ id: a.id, name: a.name, group: groupOf(a.id), x: a.x, y: a.y, magnitude: a.magnitude, peak_layer: a.peak_layer, kind })
      }
    }
    return out
  }, [data])

  if (error) return <div style={{ padding: '24px', color: '#ef4444' }}>Failed to load PCA: {error}</div>
  if (!data)  return <div style={{ padding: '24px', color: 'var(--text-muted)' }}>Loading direction embedding…</div>

  const active = VIEWS.find(v => v.id === view)!

  return (
    <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '16px' }}>
      <div style={{ marginBottom: '12px' }}>
        <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text)' }}>
          Direction representations · {vecs.length} vectors · pick a geometry
        </div>
        <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '2px' }}>
          same underlying PCA vectors, five ways of drawing them — solid = hard, dashed/redirect where applicable
        </div>
      </div>

      {/* the long list of radio buttons */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginBottom: '16px' }}>
        {VIEWS.map(v => (
          <label key={v.id} style={{
            display: 'flex', alignItems: 'baseline', gap: '8px', cursor: 'pointer',
            padding: '7px 10px', borderRadius: 'var(--radius)',
            background: view === v.id ? 'var(--surface-3)' : 'transparent',
            border: `1px solid ${view === v.id ? 'var(--border-2)' : 'transparent'}`,
          }}>
            <input type="radio" name="direction-view" value={v.id} checked={view === v.id} onChange={() => setView(v.id)} />
            <span style={{ fontSize: '13px', color: 'var(--text)', fontWeight: view === v.id ? 600 : 400 }}>{v.label}</span>
            <span style={{ fontSize: '11px', color: 'var(--text-dim)' }}>— {v.hint}</span>
          </label>
        ))}
      </div>

      <div style={{ borderTop: '1px solid var(--border)', paddingTop: '12px' }}>
        <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '8px' }}>{active.label}</div>
        {view === 'arrows'    && <ArrowsView vecs={vecs} />}
        {view === 'parallel'  && <ParallelView vecs={vecs} />}
        {view === 'radar'     && <RadarView vecs={vecs} />}
        {view === 'heatmap'   && <HeatmapView vecs={vecs} />}
        {view === 'scatter'   && <ScatterView vecs={vecs} />}
      </div>
    </div>
  )
}
