interface ModeRadioProps {
  value: string
  label: string
  selected: string
  onChange: (value: string) => void
}

export const ModeRadio = ({ value, label, selected, onChange }: ModeRadioProps) => (
  <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', color: 'var(--text-dim)' }}>
    <input
      type="radio"
      name="mode"
      value={value}
      checked={selected === value}
      onChange={() => onChange(value)}
    />
    {label}
  </label>
)
