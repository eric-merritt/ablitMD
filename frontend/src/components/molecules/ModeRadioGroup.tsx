import { ModeRadio } from '../atoms/ModeRadio'

interface ModeRadioGroupProps {
  selected: string
  onChange: (mode: string) => void
}

const MODES = [
  { value: 'non_thinking', label: 'Non-Thinking' },
  { value: 'thinking',     label: 'Thinking' },
  { value: 'both',         label: 'Both' },
]

const SectionLabel = () => (
  <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '10px' }}>
    Mode
  </div>
)

export const ModeRadioGroup = ({ selected, onChange }: ModeRadioGroupProps) => (
  <div>
    <SectionLabel />
    <div style={{ display: 'flex', gap: '20px' }}>
      {MODES.map(m => (
        <ModeRadio key={m.value} value={m.value} label={m.label} selected={selected} onChange={onChange} />
      ))}
    </div>
  </div>
)
