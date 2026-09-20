import type { AuditSummary } from '../../types/ablation'

export const SavedAuditStats = ({ audit }: { audit: AuditSummary }) => (
  <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
    {audit.n_refused}/{audit.n_trials} refused · {audit.recipe_master ?? 'no recipe'}
  </span>
)
