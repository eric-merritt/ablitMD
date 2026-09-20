import type { AuditRecord } from '../../types/ablation'
import { CardStyle } from '../atoms/CardStyle'
import { SummaryLabel } from './SummaryLabel'
import { SummaryCount } from './SummaryCount'

export const SummaryBar = ({ latest }: { latest: AuditRecord }) => (
  <div style={{ ...CardStyle, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
    <SummaryLabel n={latest.trials.length} />
    <SummaryCount nRefused={latest.trials.filter(t => t.refused).length} nTotal={latest.trials.length} />
  </div>
)
