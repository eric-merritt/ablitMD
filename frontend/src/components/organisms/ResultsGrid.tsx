import { useState } from 'react'
import { ModelResultsRow } from '../molecules/ModelResultsRow'
import { GROUPS, GROUP_LABELS } from '../../types/categories'
import type { Run } from '../../types/run'

interface ResultsGridProps {
  run: Run
  modelNames: Record<string, string>
}

export const ResultsGrid = ({ run, modelNames }: ResultsGridProps) => {
  const [visibleGroups, setVisibleGroups] = useState<Set<string>>(new Set(GROUPS))

  const toggleGroup = (groupId: string) =>
    setVisibleGroups(prev => {
      const next = new Set(prev)
      next.has(groupId) ? next.delete(groupId) : next.add(groupId)
      return next
    })

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '10px' }}>
          Visible Groups
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px' }}>
          {GROUPS.map(groupId => (
            <label key={groupId} style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer', color: 'var(--text-dim)', fontSize: '13px' }}>
              <input
                type="checkbox"
                checked={visibleGroups.has(groupId)}
                onChange={() => toggleGroup(groupId)}
              />
              {GROUP_LABELS[groupId]}
            </label>
          ))}
        </div>
      </div>
      {run.models.map(modelId => (
        <ModelResultsRow
          key={modelId}
          modelId={modelId}
          modelName={modelNames[modelId] ?? modelId}
          run={run}
          visibleGroups={visibleGroups}
        />
      ))}
    </div>
  )
}
