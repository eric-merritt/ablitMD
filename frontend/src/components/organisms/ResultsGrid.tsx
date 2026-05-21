import { useState } from 'react'
import { ModelResultsRow } from '../molecules/ModelResultsRow'
import { DirectionVectorChart } from './DirectionVectorChart'
import { AblationPanel } from './AblationPanel'
import { GROUPS, GROUP_LABELS } from '../../types/categories'
import type { Run } from '../../types/run'

interface ResultsGridProps {
  run: Run
  modelNames: Record<string, string>
  onBack: () => void
  onHome: () => void
}

const HomeIcon = () => (
  <svg width="13" height="13" viewBox="0 0 14 14" fill="none" style={{ display: 'inline-block', verticalAlign: '-2px', marginRight: '5px' }}>
    <path d="M1 7L7 1L13 7V13H9.5V9H4.5V13H1V7Z" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round" fill="none" />
  </svg>
)

export const ResultsGrid = ({ run, modelNames, onBack, onHome }: ResultsGridProps) => {
  const [visibleGroups, setVisibleGroups] = useState<Set<string>>(new Set(GROUPS))
  const [backHovered, setBackHovered] = useState(false)
  const [homeHovered, setHomeHovered] = useState(false)

  const toggleGroup = (groupId: string) =>
    setVisibleGroups(prev => {
      const next = new Set(prev)
      next.has(groupId) ? next.delete(groupId) : next.add(groupId)
      return next
    })

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{ flex: 1, overflow: 'auto' }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '24px' }}>
          <div style={{ marginBottom: '24px' }}>
            <DirectionVectorChart />
          </div>
          <div style={{ marginBottom: '24px' }}>
            <AblationPanel runId={run.run_id} />
          </div>
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
      </div>

      <div>
        <hr style={{ border: 'none', borderTop: '1px solid var(--border)', margin: 0 }} />
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 32px' }}>
          <span
            onClick={onBack}
            onMouseEnter={() => setBackHovered(true)}
            onMouseLeave={() => setBackHovered(false)}
            style={{
              color: backHovered ? 'var(--accent)' : 'var(--text)',
              fontSize: '19px',
              cursor: 'pointer',
              userSelect: 'none',
              transition: 'color 0.15s',
            }}
          >
            ←{' '}
            <span style={{ textDecoration: backHovered ? 'underline' : 'none' }}>Back</span>
          </span>

          <span
            onClick={onHome}
            onMouseEnter={() => setHomeHovered(true)}
            onMouseLeave={() => setHomeHovered(false)}
            style={{
              color: homeHovered ? 'var(--accent)' : 'var(--text)',
              fontSize: '19px',
              cursor: 'pointer',
              userSelect: 'none',
              transition: 'color 0.15s',
            }}
          >
            <HomeIcon />
            <span style={{ textDecoration: homeHovered ? 'underline' : 'none' }}>Home</span>
          </span>
        </div>
      </div>
    </div>
  )
}
