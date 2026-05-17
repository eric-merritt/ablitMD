import { CATEGORIES, GROUP_LABELS } from '../../types/categories'
import { CategoryDirectionChart } from './CategoryDirectionChart'
import type { ModeDirectionResult, RunPrompt } from '../../types/run'

interface GroupDrillChartProps {
  groupId: string
  modeResult: ModeDirectionResult
  prompts: RunPrompt[]
  modelId: string
  mode: string
}

export const GroupDrillChart = ({ groupId, modeResult, prompts, modelId, mode }: GroupDrillChartProps) => {
  const groupCats = CATEGORIES.filter(cat => cat.group === groupId)

  return (
    <div>
      <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-dim)', marginBottom: '10px' }}>
        {GROUP_LABELS[groupId]}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))', gap: '8px' }}>
        {groupCats.map(cat => {
          const dirResult = modeResult.per_category[cat.id]
          return dirResult ? (
            <CategoryDirectionChart
              key={cat.id}
              categoryName={cat.name}
              directionResult={dirResult}
              prompts={prompts}
              modelId={modelId}
              mode={mode}
            />
          ) : null
        })}
      </div>
    </div>
  )
}
