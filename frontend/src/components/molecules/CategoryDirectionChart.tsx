import { LineChart, Line, XAxis, YAxis, ReferenceLine, Tooltip, ResponsiveContainer } from 'recharts'
import { GROUP_COLORS } from '../../types/categories'
import type { CategoryDirectionResult, RunPrompt } from '../../types/run'

interface CategoryDirectionChartProps {
  categoryName: string
  directionResult: CategoryDirectionResult
  prompts: RunPrompt[]
  modelId: string
  mode: string
}

export const CategoryDirectionChart = ({
  categoryName,
  directionResult,
  prompts,
  modelId,
  mode,
}: CategoryDirectionChartProps) => {
  const { similarity_per_prompt } = directionResult
  const promptKeys = Object.keys(similarity_per_prompt)
  const numLayers = promptKeys.length > 0 ? similarity_per_prompt[promptKeys[0]].length : 0

  const chartData = Array.from({ length: numLayers }, (_, layerIndex) => {
    const point: Record<string, number> = { layer: layerIndex }
    promptKeys.forEach(key => { point[key] = similarity_per_prompt[key][layerIndex] })
    return point
  })

  const promptByKey = (key: string) =>
    prompts.find(p => p.model_results[modelId]?.[mode]?.hidden_states_key === key)

  return (
    <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '12px' }}>
      <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-dim)', marginBottom: '8px' }}>
        {categoryName}
      </div>
      <ResponsiveContainer width="100%" height={140}>
        <LineChart data={chartData} margin={{ top: 4, right: 4, bottom: 4, left: -20 }}>
          <XAxis dataKey="layer" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
          <YAxis domain={[-1, 1]} tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
          <ReferenceLine y={0} stroke="var(--border)" />
          <Tooltip
            contentStyle={{ background: 'var(--surface-2)', border: '1px solid var(--border)', borderRadius: '4px', fontSize: '11px' }}
            labelStyle={{ color: 'var(--text-muted)' }}
          />
          {promptKeys.map(key => {
            const prompt = promptByKey(key)
            const refused = prompt?.model_results[modelId]?.[mode]?.refused ?? false
            const groupColor = GROUP_COLORS[prompt?.category_group ?? ''] ?? '#888'
            return (
              <Line
                key={key}
                type="monotone"
                dataKey={key}
                stroke={groupColor}
                strokeWidth={1.5}
                strokeDasharray={refused ? undefined : '4 2'}
                dot={false}
              />
            )
          })}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
