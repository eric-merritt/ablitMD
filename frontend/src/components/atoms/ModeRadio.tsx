interface ModeRadioProps {
  value: string
  label: string
  selected: string
  onChange: (value: string) => void
}

export const ModeRadio = ({ value, label, selected, onChange }: ModeRadioProps) => (
  <label style={{
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    cursor: 'pointer',
    color: 'var(--text-dim)',
    background: 'var(--surface-3)',
    border: '1px solid var(--border-2)',
    borderRadius: 'var(--radius)',
    padding: '7px 10px',
    userSelect: 'none',
  }}>
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
