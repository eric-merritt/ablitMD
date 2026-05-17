import { useState } from 'react'
import { ModelCheckbox } from '../atoms/ModelCheckbox'
import { GROUP_LABELS, GROUPS, catsByGroup } from '../../types/categories'

interface CategoryListProps {
  selectedCategories: Set<string>
  onCategoryToggle: (categoryId: string, checked: boolean) => void
  onGroupToggle: (groupId: string, checked: boolean) => void
}

const SectionLabel = () => (
  <div style={{ fontSize: '13px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '10px', background: 'var(--surface-3)', border: '1px solid var(--border-2)', borderRadius: 'var(--radius)', padding: '7px 10px', color: 'var(--text-muted)' }}>
    Categories
  </div>
)

export const CategoryList = ({ selectedCategories, onCategoryToggle, onGroupToggle }: CategoryListProps) => {
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set())

  const toggleGroup = (groupId: string) => setExpandedGroups(prev => {
    const next = new Set(prev)
    next.has(groupId) ? next.delete(groupId) : next.add(groupId)
    return next
  })

  const isGroupChecked = (groupId: string) =>
    catsByGroup(groupId).every(cat => selectedCategories.has(cat.id))

  return (
    <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
      <li><SectionLabel /></li>
      <li>
        <ul style={{ paddingLeft: '16px', listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '4px' }}>
          {GROUPS.map(groupId => (
            <li key={groupId}>
              <ModelCheckbox
                label={GROUP_LABELS[groupId]}
                checked={isGroupChecked(groupId)}
                onChange={checked => onGroupToggle(groupId, checked)}
                onExpand={() => toggleGroup(groupId)}
                expanded={expandedGroups.has(groupId)}
                variant="group"
              />
              {expandedGroups.has(groupId) && (
                <ul style={{ paddingLeft: '16px', listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '4px', marginTop: '4px' }}>
                  {catsByGroup(groupId).map(cat => (
                    <li key={cat.id}>
                      <ModelCheckbox
                        label={cat.name}
                        checked={selectedCategories.has(cat.id)}
                        onChange={checked => onCategoryToggle(cat.id, checked)}
                        variant="category"
                      />
                    </li>
                  ))}
                </ul>
              )}
            </li>
          ))}
        </ul>
      </li>
    </ul>
  )
}
