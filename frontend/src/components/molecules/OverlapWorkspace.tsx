import { useCallback, useEffect, useMemo, useState } from 'react'
import { directionOverlap } from '../../api/ablation'
import type { DirectionOverlapResponse } from '../../types/ablation'
import { CATEGORIES } from '../../types/categories'
import { SectionTitle } from '../atoms/SectionTitle'
import { ErrorText } from '../atoms/ErrorText'
import { OverlapLegend } from './OverlapLegend'
import { OverlapNote } from './OverlapNote'
import { OverlapStats } from './OverlapStats'

interface OverlapWorkspaceProps {
  run: { run_id: string; sequence?: { model: string; mode: string }[] }
  selected: Set<string>
}

const CATEGORY_NAMES: Record<string, string> = {}
for (const cat of CATEGORIES) CATEGORY_NAMES[cat.id] = cat.name
const labelOf = (id: string): string => CATEGORY_NAMES[id] ?? id

// --- Colors ------------------------------------------------------------------
// Experiment arrows are white. Recipe-only preview arrows are red.
// When experiments + recipe are shown together, experiment arrows stay white
// and a thin red overlay is drawn on top for comparison.
const EXPERIMENT_ARROW = 'rgba(255,255,255,0.95)'   // white
const RECIPE_ARROW   = 'rgb(255,0,0)'                // bright red

// --- SVG geometry -----------------------------------------------------------

const W = 1170
const H = 676
const MARGIN = { top: 60, right: 60, bottom: 60, left: 60 }

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

// --- Component --------------------------------------------------------------

export const OverlapWorkspace = ({ run, selected }: OverlapWorkspaceProps) => {
  const [data, setData] = useState<DirectionOverlapResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Recipe-only = __recipe__ sentinel present with no real audit paths.
  const isRecipeOnly = selected.has('__recipe__') &&
    ![...selected].some(p => p !== '__recipe__')

  // Fetch arrows on mount (shows recipe directions immediately), re-fetch when selection changes.
  useEffect(() => {
    const step = run.sequence?.[0]
    if (!step) { setError('No model/mode in run sequence'); return }
    let cancelled = false
    setLoading(true)
    setError(null)

    // Filter out the __recipe__ sentinel — it means "show arrows from recipe, no experiments."
    const realPaths = [...selected].filter(p => p !== '__recipe__')

    directionOverlap({
      run_id: run.run_id,
      model_id: step.model,
      mode: step.mode,
      experiments: realPaths.map(path => ({ path })),
    })
      .then(res => { if (!cancelled) setData(res) })
      .catch(err => { if (!cancelled) setError(err instanceof Error ? err.message : String(err)) })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [selected, run.run_id, run.sequence])

  const arrows = data?.arrows ?? {}
  const arrowList = useMemo(() => Object.values(arrows), [arrows])
  const layout = fitLayout(arrowList)

  // In recipe-only mode (no audit yet), ALL arrows are "recipe directions" —
  // render them red. Otherwise split by refused/ok from experiments.
  const allArrowCats = useMemo(() => Object.keys(arrows), [arrows])

  const { refusedCats, okCats } = useMemo(() => {
    if (isRecipeOnly) {
      return { refusedCats: [] as string[], okCats: allArrowCats }
    }
    const refused = new Set<string>()
    const ok = new Set<string>()
    for (const exp of data?.experiments ?? []) {
      for (const c of exp.refused_categories) refused.add(c)
      for (const c of exp.ok_categories) ok.add(c)
    }
    for (const c of refused) ok.delete(c)

    // Always include every arrow category — experiment data just tags which are refused.
    // Categories not mentioned by any experiment still render (as non-refused).
    const allCats = new Set(allArrowCats)
    for (const c of allCats) {
      if (!refused.has(c) && !ok.has(c)) ok.add(c)
    }

    return { refusedCats: [...refused], okCats: [...ok] }
  }, [data, isRecipeOnly, allArrowCats])

  // Draw a single arrow line + label. `stroke` controls the color.
  const renderArrow = (cat: string, stroke: string, strokeWidth: number) => {
    if (!layout) return null
    const arrow = arrows[cat]
    if (!arrow) return null
    const tipX = layout.cx
    const tipY = layout.cy
    const dirX = layout.toX(arrow.x) - tipX
    const dirY = layout.toY(arrow.y) - tipY

    return (
      <g key={cat}>
        <line x1={tipX} y1={tipY} x2={tipX + dirX} y2={tipY + dirY}
          stroke={stroke} strokeWidth={strokeWidth} />
        <text x={tipX + dirX * 1.08} y={tipY + dirY * 1.08} fontSize={13}
          fill="var(--text-dim)" textAnchor="middle">{labelOf(cat)}</text>
      </g>
    )
  }

  // Collect all arrows to render, choosing the right color per mode.
  const arrowPaths: React.ReactNode[] = []
  for (const cat of [...refusedCats, ...okCats]) {
    if (isRecipeOnly) {
      // Recipe-only mode → red arrows
      arrowPaths.push(renderArrow(cat, RECIPE_ARROW, 3))
    } else {
      // Experiment mode → white arrows
      arrowPaths.push(renderArrow(cat, EXPERIMENT_ARROW, 2))
    }
  }

  // Recipe-direction overlay: thin red lines on top when experiments are selected.
  const recipeOverlay: React.ReactNode[] = []
  if (!isRecipeOnly && layout) {
    for (const cat of allArrowCats) {
      const arrow = arrows[cat]
      if (!arrow) continue
      const tipX = layout.cx
      const tipY = layout.cy
      const dirX = layout.toX(arrow.x) - tipX
      const dirY = layout.toY(arrow.y) - tipY
      recipeOverlay.push(
        <g key={'recipe-overlay-' + cat} style={{ pointerEvents: 'none' }}>
          <line x1={tipX} y1={tipY} x2={tipX + dirX} y2={tipY + dirY}
            stroke={RECIPE_ARROW} strokeWidth={1.5} />
        </g>,
      )
    }
  }

  return (
    <div id="overlap-workspace" style={{ background: 'var(--bg)', borderRadius: 'var(--radius)', border: '1px solid var(--border)', padding: '16px' }}>
      <SectionTitle>Direction-overlap workspace</SectionTitle>
      {error && <ErrorText message={error} />}
      {loading && <div style={{ fontSize: '12px', color: 'var(--text-dim)' }}>Computing overlap…</div>}

      {data && layout && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', width: '100%' }}>
          <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', minWidth: '780px', height: 676, background: 'var(--surface-2)', borderRadius: 'var(--radius)', border: '1px solid var(--border)' }}>
            {/* axes */}
            <line x1={MARGIN.left} y1={layout.cy} x2={W - MARGIN.right} y2={layout.cy} stroke="var(--border, #444)" strokeWidth={0.5} />
            <line x1={layout.cx} y1={MARGIN.top} x2={layout.cx} y2={H - MARGIN.bottom} stroke="var(--border, #444)" strokeWidth={0.5} />

            {/* Arrow lines */}
            {arrowPaths}

            {/* Recipe overlay (red) when experiments are selected */}
            {!isRecipeOnly && recipeOverlay}
          </svg>

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px' }}>
            <div style={{ maxWidth: '30%' }}><OverlapStats data={data} /></div>
            <div style={{ maxWidth: '30%' }}><OverlapLegend refusedCats={refusedCats} okCats={okCats} /></div>
          </div>
        </div>
      )}
    </div>
  )
}
