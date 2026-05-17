import { useState } from 'react'
import type { CSSProperties } from 'react'
import { CategoryDirectionChart } from './CategoryDirectionChart'
import { CATEGORIES } from '../../types/categories'
import type { Run } from '../../types/run'

interface ModelResultsRowProps {
  modelId: string
  modelName: string
  run: Run
  visibleGroups: Set<string>
}

const tabBtnStyle = (active: boolean): CSSProperties => ({
  padding: '4px 12px',
  borderRadius: '4px',
  border: '1px solid var(--border)',
  background: active ? 'var(--accent)' : 'var(--surface)',
  color: active ? '#fff' : 'var(--text-dim)',
  fontSize: '12px',
  fontWeight: 600,
  cursor: 'pointer',
})

export const ModelResultsRow = ({ modelId, modelName, run, visibleGroups }: ModelResultsRowProps) => {
  const [activeTab, setActiveTab] = useState('non_thinking')
  const [expanded, setExpanded] = useState(false)

  const modelResults = run.direction_results?.[modelId] ?? {}
  const availableModes = Object.keys(modelResults)

  const TabBar = () => (
    <div style={{ display: 'flex', gap: '4px', marginBottom: '12px' }}>
      {availableModes.map(mode => (
        <button key={mode} onClick={() => setActiveTab(mode)} style={tabBtnStyle(activeTab === mode)}>
          {mode === 'non_thinking' ? 'Non-Thinking' : 'Thinking'}
        </button>
      ))}
    </div>
  )

  const ChartGrid = ({ mode }: { mode: string }) => {
    const modeResult = modelResults[mode]
    if (!modeResult) return <div style={{ color: 'var(--text-muted)', fontSize: '13px' }}>No data</div>

    const visibleCats = CATEGORIES.filter(
      cat => visibleGroups.has(cat.group) && modeResult.per_category[cat.id]
    )

    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))', gap: '8px' }}>
        {visibleCats.map(cat => (
          <CategoryDirectionChart
            key={cat.id}
            categoryName={cat.name}
            directionResult={modeResult.per_category[cat.id]}
            prompts={run.prompts}
            modelId={modelId}
            mode={mode}
          />
        ))}
      </div>
    )
  }

  return (
    <div style={{ marginBottom: '32px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
        <h3 style={{ color: 'var(--text)', fontSize: '15px', fontWeight: 600 }}>{modelName}</h3>
        <button
          onClick={() => setExpanded(prev => !prev)}
          style={{ padding: '4px 10px', borderRadius: '4px', border: '1px solid var(--border)', background: 'var(--surface)', color: 'var(--text-dim)', fontSize: '12px', cursor: 'pointer' }}
        >
          {expanded ? 'Collapse' : 'Expand'}
        </button>
      </div>
      {expanded ? (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
          {availableModes.map(mode => (
            <div key={mode}>
              <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '8px' }}>
                {mode === 'non_thinking' ? 'Non-Thinking' : 'Thinking'}
              </div>
              <ChartGrid mode={mode} />
            </div>
          ))}
        </div>
      ) : (
        <div>
          <TabBar />
          <ChartGrid mode={activeTab} />
        </div>
      )}
    </div>
  )
}
