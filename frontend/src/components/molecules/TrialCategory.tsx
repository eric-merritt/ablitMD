import { CATEGORIES } from '../../types/categories'

const CATEGORY_NAMES: Record<string, string> = {}
for (const cat of CATEGORIES) CATEGORY_NAMES[cat.id] = cat.name
const labelOf = (id: string): string => CATEGORY_NAMES[id] ?? id

export const TrialCategory = ({ id }: { id: string }) => (
  <span style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
    {labelOf(id)}
  </span>
)
