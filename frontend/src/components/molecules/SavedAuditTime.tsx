import type { AuditSummary } from '../../types/ablation'

export const SavedAuditTime = ({ audit }: { audit: AuditSummary }) => (
  <span style={{ fontSize: '11px', color: 'var(--text-dim)' }}>
    {new Date(audit.created_at).toLocaleString()}
  </span>
)
