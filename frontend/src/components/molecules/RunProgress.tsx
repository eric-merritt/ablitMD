interface RunProgressProps {
  modelName: string
  mode: string
  currentIndex: number
  total: number
}

const modeLabel = (mode: string) => mode === 'non_thinking' ? 'Non-Thinking' : 'Thinking'

export const RunProgress = ({ modelName, mode, currentIndex, total }: RunProgressProps) => (
  <div style={{ padding: '12px 20px', background: 'var(--surface)', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '13px' }}>
    <span style={{ color: 'var(--text-dim)' }}>
      <strong style={{ color: 'var(--text)' }}>{modelName}</strong>
      {' | '}
      <strong style={{ color: 'var(--text)' }}>{modeLabel(mode)}</strong>
    </span>
    <span style={{ color: 'var(--text-muted)' }}>
      Prompt <strong style={{ color: 'var(--accent)' }}>{currentIndex + 1}</strong> / {total}
    </span>
  </div>
)
