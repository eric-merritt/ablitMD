interface ModelCheckboxProps {
  label: string
  checked: boolean
  onChange: (checked: boolean) => void
  onExpand?: () => void
  expanded?: boolean
  variant?: 'model' | 'group' | 'category'
}

const BG = {
  model:    'var(--surface)',
  group:    'var(--surface-2)',
  category: 'var(--surface-3)',
}

const BORDER = {
  model:    'var(--border)',
  group:    'var(--border)',
  category: 'var(--border-2)',
}

export const ModelCheckbox = ({
  label,
  checked,
  onChange,
  onExpand,
  expanded = false,
  variant = 'model',
}: ModelCheckboxProps) => (
  <div style={{
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    border: `1px solid ${BORDER[variant]}`,
    borderRadius: 'var(--radius)',
    padding: '7px 4px 7px 10px',
    background: BG[variant],
    userSelect: 'none',
  }}>
    <input
      type="checkbox"
      checked={checked}
      onChange={evt => onChange(evt.target.checked)}
    />
    <span style={{ flex: 1, color: variant === 'model' ? 'var(--text)' : 'var(--text-dim)' }}>
      {label}
    </span>
    {onExpand && (
      <button
        type="button"
        onClick={onExpand}
        style={{ fontSize: '16px', lineHeight: '1', color: 'var(--text-muted)', padding: '0 4px', cursor: 'pointer', borderRadius: '4px', flexShrink: 0, background: 'none', border: 'none' }}
      >
        {expanded ? '−' : '+'}
      </button>
    )}
  </div>
)
