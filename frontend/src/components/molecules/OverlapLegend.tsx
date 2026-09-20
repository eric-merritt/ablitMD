import { CATEGORIES } from '../../types/categories'

const CATEGORY_NAMES: Record<string, string> = {}
for (const cat of CATEGORIES) CATEGORY_NAMES[cat.id] = cat.name
const labelOf = (id: string): string => CATEGORY_NAMES[id] ?? id

export const OverlapLegend = ({ refusedCats, okCats }: { refusedCats: string[]; okCats: string[] }) => (
  <div style={{ display: 'flex', gap: '24px', marginTop: '10px', fontSize: '12px' }}>
    <div>
      <span style={{ color: 'var(--danger)', fontWeight: 600 }}>{`Refused (${refusedCats.length})`}</span>{' '}
      <span style={{ color: 'var(--text-dim)' }}>{refusedCats.map(labelOf).join(', ') || '—'}</span>
    </div>
    <div>
      <span style={{ color: 'var(--ok)', fontWeight: 600 }}>{`Non-refused (${okCats.length})`}</span>{' '}
      <span style={{ color: 'var(--text-dim)' }}>{okCats.map(labelOf).join(', ') || '—'}</span>
    </div>
  </div>
)
