import { ModelCheckbox } from '../atoms/ModelCheckbox'
import type { LLM } from '../../types/model'

interface ModelCheckboxListProps {
  models: LLM[]
  selectedModels: Set<string>
  onModelToggle: (modelId: string, checked: boolean) => void
}

const SectionLabel = () => (
  <div style={{ fontSize: '13px', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '10px', background: 'var(--surface-3)', border: '1px solid var(--border-2)', borderRadius: 'var(--radius)', padding: '7px 10px', color: 'var(--text-muted)' }}>
    Models
  </div>
)

export const ModelCheckboxList = ({ models, selectedModels, onModelToggle }: ModelCheckboxListProps) => (
  <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
    <li><SectionLabel /></li>
    <li>
      <ul style={{ paddingLeft: '16px', listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '4px' }}>
        {models.map(model => (
          <li key={model.modelId}>
            <ModelCheckbox
              label={model.name}
              checked={selectedModels.has(model.modelId)}
              onChange={checked => onModelToggle(model.modelId, checked)}
              variant="group"
            />
          </li>
        ))}
      </ul>
    </li>
  </ul>
)
