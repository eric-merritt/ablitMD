import type { AuditSummary } from '../../types/ablation'
import { SavedAuditTime } from './SavedAuditTime'
import { SavedAuditStats } from './SavedAuditStats'

export const ExperimentRow = ({ audit, checked, onToggle }: { audit: AuditSummary; checked: boolean; onToggle: () => void }) => (
  <label style={{ display: 'flex', alignItems: 'flex-start', gap: '8px', fontSize: '12px', color: 'var(--text-dim)', cursor: 'pointer' }}>
    <input type="checkbox" checked={checked} onChange={onToggle} />
    <span style={{ flex: 1 }}>
      <SavedAuditTime audit={audit} /><br />
      <SavedAuditStats audit={audit} />
    </span>
  </label>
)
