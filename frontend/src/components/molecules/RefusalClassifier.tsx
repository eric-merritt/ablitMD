import type { RefusalMode } from '../../types/run'
import { RefusalModeRadio } from '../atoms/RefusalModeRadio'

type Selection = RefusalMode | 'none' | null

interface RefusalClassifierProps {
  disabled: boolean
  selected: Selection
  onChange: (value: RefusalMode | 'none') => void
}

const REFUSED_MODES: { value: RefusalMode; label: string }[] = [
  { value: 'hard',       label: 'Hard' },
  { value: 'redirect',   label: 'Redirect' },
  { value: 'disclaimer', label: 'Disclaimer' },
]

export const RefusalClassifier = ({ disabled, selected, onChange }: RefusalClassifierProps) => (
  <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
    <button
      disabled={disabled}
      onClick={() => onChange('none')}
      style={{
        padding: '10px 20px',
        borderRadius: 'var(--radius)',
        border: `1px solid ${selected === 'none' ? '#16a34a' : 'var(--border)'}`,
        background: selected === 'none' ? 'rgba(22,163,74,0.08)' : 'transparent',
        color: '#16a34a',
        fontWeight: 600,
        alignSelf: 'flex-start',
        opacity: disabled ? 0.4 : 1,
        cursor: disabled ? 'not-allowed' : 'pointer',
      }}
    >
      Not Refused
    </button>
    <div style={{ borderTop: '1px solid var(--border)', paddingTop: '12px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
      <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
        Refused
      </div>
      {REFUSED_MODES.map(m => (
        <RefusalModeRadio
          key={m.value}
          value={m.value}
          label={m.label}
          selected={disabled || selected === 'none' ? null : selected as RefusalMode | null}
          onChange={onChange}
        />
      ))}
    </div>
  </div>
)
