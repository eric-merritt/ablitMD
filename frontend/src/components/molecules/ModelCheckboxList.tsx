import { useState } from 'react'
import { ModelCheckbox } from '../atoms/ModelCheckbox'
import { CATEGORIES, GROUP_LABELS, GROUPS, catsByGroup } from '../../types/categories'
import type { LLM } from '../../types/model'

interface ModelCheckboxListProps {
  models: LLM[]
  selectedModels: Set<string>
  selectedCategories: Set<string>
  onModelToggle: (modelId: string, checked: boolean) => void
  onCategoryToggle: (categoryId: string, checked: boolean) => void
  onGroupToggle: (groupId: string, checked: boolean) => void
}

const SectionLabel = () => (
  <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '10px' }}>
    Models &amp; Categories
  </div>
)

export const ModelCheckboxList = ({
  models,
  selectedModels,
  selectedCategories,
  onModelToggle,
  onCategoryToggle,
  onGroupToggle,
}: ModelCheckboxListProps) => {
  const [expandedModels, setExpandedModels] = useState<Set<string>>(new Set())
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set())

  const toggleModel = (modelId: string) => setExpandedModels(prev => {
    const next = new Set(prev)
    next.has(modelId) ? next.delete(modelId) : next.add(modelId)
    return next
  })

  const toggleGroup = (groupId: string) => setExpandedGroups(prev => {
    const next = new Set(prev)
    next.has(groupId) ? next.delete(groupId) : next.add(groupId)
    return next
  })

  const isGroupChecked = (groupId: string) =>
    catsByGroup(groupId).every(cat => selectedCategories.has(cat.id))

  return (
    <div>
      <SectionLabel />
      <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
        {models.map(model => (
          <div key={model.modelId}>
            <ModelCheckbox
              label={model.name}
              checked={selectedModels.has(model.modelId)}
              onChange={checked => onModelToggle(model.modelId, checked)}
              onExpand={() => toggleModel(model.modelId)}
              expanded={expandedModels.has(model.modelId)}
              variant="model"
            />
            {expandedModels.has(model.modelId) && (
              <div style={{ marginLeft: '20px', display: 'flex', flexDirection: 'column', gap: '4px', marginTop: '4px' }}>
                {GROUPS.map(groupId => (
                  <div key={groupId}>
                    <ModelCheckbox
                      label={GROUP_LABELS[groupId]}
                      checked={isGroupChecked(groupId)}
                      onChange={checked => onGroupToggle(groupId, checked)}
                      onExpand={() => toggleGroup(groupId)}
                      expanded={expandedGroups.has(groupId)}
                      variant="group"
                    />
                    {expandedGroups.has(groupId) && (
                      <div style={{ marginLeft: '20px', display: 'flex', flexDirection: 'column', gap: '4px', marginTop: '4px' }}>
                        {catsByGroup(groupId).map(cat => (
                          <ModelCheckbox
                            key={cat.id}
                            label={cat.name}
                            checked={selectedCategories.has(cat.id)}
                            onChange={checked => onCategoryToggle(cat.id, checked)}
                            variant="category"
                          />
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
