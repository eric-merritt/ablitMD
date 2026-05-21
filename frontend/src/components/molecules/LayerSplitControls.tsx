interface LayerSplitControlsProps {
  onset: number
  split: number
  lastLayer: number
  onChange: (next: { onset: number; split: number }) => void
}

const NumberField = ({ label, value, min, max, onCommit }: {
  label: string; value: number; min: number; max: number; onCommit: (next: number) => void
}) => (
  <label style={{ display: 'flex', flexDirection: 'column', gap: '3px', fontSize: '11px', color: 'var(--text-muted)' }}>
    {label}
    <input
      type="number"
      value={value}
      min={min}
      max={max}
      onChange={event => {
        const next = Number(event.target.value)
        if (Number.isFinite(next) && next >= min && next <= max) onCommit(next)
      }}
      style={{ width: '70px', padding: '4px 6px', background: 'var(--surface-2)', color: 'var(--text)', border: '1px solid var(--border)', borderRadius: '4px' }}
    />
  </label>
)

export const LayerSplitControls = ({ onset, split, lastLayer, onChange }: LayerSplitControlsProps) => (
  <div style={{ display: 'flex', gap: '16px', alignItems: 'flex-end' }}>
    <NumberField label="onset layer" value={onset} min={0} max={split}
      onCommit={next => onChange({ onset: next, split })} />
    <NumberField label="split layer" value={split} min={onset} max={lastLayer}
      onCommit={next => onChange({ onset, split: next })} />
  </div>
)
