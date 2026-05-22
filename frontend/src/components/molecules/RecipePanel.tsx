import type { SlimRecipe } from '../../types/ablation'

interface RecipePanelProps {
  recipe: SlimRecipe
  onset: number
  split: number
}

export const RecipePanel = ({ recipe, onset, split }: RecipePanelProps) => {
  const modeNames = Object.keys(recipe.modes)
  return (
    <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '16px' }}>
      <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text)', marginBottom: '8px' }}>
        Recipe · {modeNames.join(' + ')}
      </div>
      <div style={{ fontSize: '11px', color: 'var(--text-dim)' }}>
        Phase A (shared) layers {onset}–{split} · Phase B (per-category) layers {split}–{recipe.last_layer}
      </div>
    </div>
  )
}
