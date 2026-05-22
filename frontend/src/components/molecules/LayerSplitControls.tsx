import { useState, useEffect } from 'react'
import type { RecipeParams } from '../../types/ablation'

interface LayerSplitControlsProps {
  params: RecipeParams
  lastLayer: number
  onChange: (next: RecipeParams) => void
}

const NumberField = ({ label, value, min, max, onCommit }: {
  label: string; value: number; min: number; max: number; onCommit: (next: number) => void
}) => {
  const [draft, setDraft] = useState(String(value))

  useEffect(() => { setDraft(String(value)) }, [value])

  const commit = () => {
    const next = Number(draft)
    if (Number.isFinite(next) && next >= min && next <= max) onCommit(next)
    else setDraft(String(value))
  }

  return (
    <label style={{ display: 'flex', flexDirection: 'column', gap: '3px', fontSize: '11px', color: 'var(--text-muted)' }}>
      {label}
      <input
        type="number"
        value={draft}
        min={min}
        max={max}
        onChange={event => setDraft(event.target.value)}
        onBlur={commit}
        onKeyDown={event => { if (event.key === 'Enter') commit() }}
        style={{ width: '70px', padding: '4px 6px', background: 'var(--surface-2)', color: 'var(--text)', border: '1px solid var(--border)', borderRadius: '4px' }}
      />
    </label>
  )
}

const FactorField = ({ label, value, onCommit }: {
  label: string; value: number; onCommit: (next: number) => void
}) => (
  <label style={{ display: 'flex', flexDirection: 'column', gap: '3px', fontSize: '11px', color: 'var(--text-muted)', flex: 1, minWidth: '120px' }}>
    <span>{label} <strong style={{ color: 'var(--text)' }}>{value.toFixed(2)}</strong></span>
    <input
      type="range"
      min={0}
      max={3}
      step={0.05}
      value={value}
      onChange={event => onCommit(Number(event.target.value))}
      style={{ width: '100%' }}
    />
  </label>
)

export const LayerSplitControls = ({ params, lastLayer, onChange }: LayerSplitControlsProps) => (
  <div style={{ display: 'flex', gap: '16px', alignItems: 'flex-end', flexWrap: 'wrap' }}>
    <NumberField label="onset layer" value={params.onset} min={0} max={params.split}
      onCommit={next => onChange({ ...params, onset: next })} />
    <NumberField label="split layer" value={params.split} min={params.onset} max={lastLayer}
      onCommit={next => onChange({ ...params, split: next })} />
    <FactorField label="factor A" value={params.factorA}
      onCommit={next => onChange({ ...params, factorA: next })} />
    <FactorField label="factor B" value={params.factorB}
      onCommit={next => onChange({ ...params, factorB: next })} />
  </div>
)
