import { useState, useEffect } from 'react'

interface CategoryFactorSlidersProps {
  categories: string[]
  factorAByCategory: Record<string, number>
  masterFactor: number
  onChange: (category: string, value: number) => void
}

const CategoryRow = ({ category, value, overridden, onChange }: {
  category: string; value: number; overridden: boolean; onChange: (next: number) => void
}) => {
  const [draft, setDraft] = useState(value.toFixed(2))
  useEffect(() => { setDraft(value.toFixed(2)) }, [value])

  const commit = () => {
    const next = Number(draft)
    if (Number.isFinite(next) && next >= 0 && next <= 3) onChange(next)
    else setDraft(value.toFixed(2))
  }

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '120px 1fr 58px', gap: '8px', alignItems: 'center' }}>
      <span style={{ fontSize: '11px', color: overridden ? 'var(--accent)' : 'var(--text-muted)',
                     fontWeight: overridden ? 600 : 400, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        { category }{ overridden ? ' *' : '' }
      </span>
      <input type="range" min={0} max={3} step={0.05} value={value}
        onChange={event => onChange(Number(event.target.value))} style={{ width: '100%' }} />
      <input type="number" min={0} max={3} step={0.05} value={draft}
        onChange={event => setDraft(event.target.value)}
        onBlur={commit}
        onKeyDown={event => { if (event.key === 'Enter') commit() }}
        style={{ width: '58px', padding: '3px 5px', background: 'var(--surface-2)', color: 'var(--text)',
                 border: '1px solid var(--border)', borderRadius: '4px', fontSize: '11px' }} />
    </div>
  )
}

const CategoryFactorHeader = () => (
  <div style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase',
                letterSpacing: '0.05em', marginBottom: '2px' }}>
    factor A · per category <span style={{ color: 'var(--accent)' }}>(* = overridden)</span>
  </div>
)

export const CategoryFactorSliders = ({ categories, factorAByCategory, masterFactor, onChange }: CategoryFactorSlidersProps) => {
  if (categories.length === 0) return null
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', paddingTop: '8px',
                  borderTop: '1px solid var(--border)' }}>
      <CategoryFactorHeader />
      { categories.map(category => (
        <CategoryRow key={ category } category={ category }
          value={ factorAByCategory[category] ?? masterFactor }
          overridden={ category in factorAByCategory }
          onChange={ next => onChange(category, next) } />
      )) }
    </div>
  )
}
