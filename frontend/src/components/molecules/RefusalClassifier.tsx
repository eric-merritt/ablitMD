import { useState } from 'react'
import { RefusalModeRadio } from '../atoms/RefusalModeRadio'
import type { RefusalMode } from '../../types/run'

interface RefusalClassifierProps {
  disabled: boolean
  onClassify: (refused: boolean, mode: RefusalMode) => void
}

const REFUSED_MODES: { value: RefusalMode; label: string }[] = [
  { value: 'hard',       label: 'Hard' },
  { value: 'redirect',   label: 'Redirect' },
  { value: 'disclaimer', label: 'Disclaimer' },
]

export const RefusalClassifier = ({ disabled, onClassify }: RefusalClassifierProps) => {
  const [selected, setSelected] = useState<RefusalMode | null>(null)

  const handleRefused = (mode: RefusalMode) => {
    setSelected(mode)
    onClassify(true, mode)
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
      <button
        disabled={disabled}
        onClick={() => { setSelected(null); onClassify(false, 'none') }}
        style={{
          padding: '10px 20px',
          borderRadius: 'var(--radius)',
          border: '1px solid #16a34a',
          background: 'transparent',
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
            selected={disabled ? null : selected}
            onChange={handleRefused}
          />
        ))}
      </div>
    </div>
  )
}
