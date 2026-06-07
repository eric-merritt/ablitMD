import type { RecipePriorAttempt, CategoryOutcome } from '../../types/ablation'

interface RecipeReuseWarningProps {
  priorAttempt: RecipePriorAttempt
  onset: number
  split: number
  factorA: number
  factorB: number
}

const stillRefused = (outcome: CategoryOutcome) => outcome.refused > 0
const complied = (outcome: CategoryOutcome) => outcome.refused === 0 && outcome.complied > 0

const CategoryList = ({ label, color, outcomes }: {
  label: string; color: string; outcomes: CategoryOutcome[]
}) => {
  if (outcomes.length === 0) return null
  return (
    <div style={{ fontSize: '11px', color: 'var(--text)' }}>
      <span style={{ color, fontWeight: 600 }}>{ label } ({ outcomes.length }): </span>
      { outcomes.map(outcome => `${outcome.category} (${outcome.refused}r/${outcome.complied}c)`).join(', ') }
    </div>
  )
}

const WarningHeader = ({ onset, split, factorA, factorB, verifiedAt }: {
  onset: number; split: number; factorA: number; factorB: number; verifiedAt: string
}) => (
  <div style={{ fontSize: '12px', fontWeight: 700, color: '#fbbf24' }}>
    ⚠ You've verified this recipe before — onset { onset } · split { split } · factorA { factorA } · factorB { factorB }
    <span style={{ fontWeight: 400, color: 'var(--text-muted)' }}> ({ new Date(verifiedAt).toLocaleString() })</span>
  </div>
)

export const RecipeReuseWarning = ({ priorAttempt, onset, split, factorA, factorB }: RecipeReuseWarningProps) => {
  const refused = priorAttempt.categories.filter(stillRefused)
  const compl = priorAttempt.categories.filter(complied)
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', padding: '10px 12px',
                  background: 'rgba(251, 191, 36, 0.08)', border: '1px solid rgba(251, 191, 36, 0.4)',
                  borderRadius: 'var(--radius)' }}>
      <WarningHeader onset={ onset } split={ split } factorA={ factorA } factorB={ factorB } verifiedAt={ priorAttempt.verified_at } />
      <CategoryList label="Still refused" color="#f87171" outcomes={ refused } />
      <CategoryList label="Complied" color="#6ee7b7" outcomes={ compl } />
      <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
        Tweak the per-category factor A sliders for the still-refused categories instead of re-running this same recipe.
      </div>
    </div>
  )
}
