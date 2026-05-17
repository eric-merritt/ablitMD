import { GROUP_LABELS } from '../../types/categories'
import type { RunPrompt } from '../../types/run'

interface PromptCardProps {
  prompt: RunPrompt
}

const Badge = ({ label, color }: { label: string; color: string }) => (
  <span style={{ padding: '2px 8px', borderRadius: '4px', fontSize: '11px', fontWeight: 600, background: color, color: '#fff', textTransform: 'uppercase' as const, letterSpacing: '0.05em' }}>
    {label}
  </span>
)

export const PromptCard = ({ prompt }: PromptCardProps) => (
  <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '20px' }}>
    <div style={{ display: 'flex', gap: '8px', marginBottom: '12px' }}>
      <Badge label={prompt.type} color={prompt.type === 'harmful' ? '#dc2626' : '#16a34a'} />
      <Badge label={GROUP_LABELS[prompt.category_group] ?? prompt.category_group} color="#4b5563" />
    </div>
    <p style={{ color: 'var(--text)', lineHeight: '1.6' }}>{prompt.text}</p>
  </div>
)
