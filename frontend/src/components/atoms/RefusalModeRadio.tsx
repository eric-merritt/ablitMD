import type { RefusalMode } from '../../types/run'

interface RefusalModeRadioProps {
  value: RefusalMode
  label: string
  selected: RefusalMode | null
  onChange: (value: RefusalMode) => void
}

export const RefusalModeRadio = ({ value, label, selected, onChange }: RefusalModeRadioProps) => (
  <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', color: 'var(--text-dim)' }}>
    <input
      type="radio"
      name="refusal_mode"
      value={value}
      checked={selected === value}
      onChange={() => onChange(value)}
    />
    {label}
  </label>
)
