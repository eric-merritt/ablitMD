import type { SlimRecipe } from '../../types/ablation'

interface RecipePanelProps {
  recipe: SlimRecipe
  onset: number
  split: number
  factorA: number
  factorB: number
  onFactorChange: (next: { factorA: number; factorB: number }) => void
}

const FactorSlider = ({ label, value, onCommit }: {
  label: string; value: number; onCommit: (next: number) => void
}) => (
  <label style={{ display: 'flex', flexDirection: 'column', gap: '4px', fontSize: '11px', color: 'var(--text-muted)' }}>
    <span>{label}: <strong style={{ color: 'var(--text)' }}>{value.toFixed(2)}</strong></span>
    <input
      type="range"
      min={0}
      max={1}
      step={0.05}
      value={value}
      onChange={event => onCommit(Number(event.target.value))}
    />
  </label>
)

export const RecipePanel = ({ recipe, onset, split, factorA, factorB, onFactorChange }: RecipePanelProps) => {
  const modeNames = Object.keys(recipe.modes)
  return (
    <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '16px' }}>
      <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text)', marginBottom: '8px' }}>
        Recipe · {modeNames.join(' + ')}
      </div>
      <div style={{ fontSize: '11px', color: 'var(--text-dim)', marginBottom: '12px' }}>
        Phase A (shared) layers {onset}–{split} · Phase B (per-category)
        layers {split}–{recipe.last_layer}
      </div>
      <div style={{ display: 'flex', gap: '24px' }}>
        <FactorSlider label="factor A (shared)" value={factorA}
          onCommit={next => onFactorChange({ factorA: next, factorB })} />
        <FactorSlider label="factor B (per-category)" value={factorB}
          onCommit={next => onFactorChange({ factorA, factorB: next })} />
      </div>
    </div>
  )
}
