import type { AuditSummary } from '../../types/ablation'
import { CardStyle } from '../atoms/CardStyle'
import { SectionTitle } from '../atoms/SectionTitle'
import { ExperimentRow } from './ExperimentRow'

interface ExperimentListProps {
  audits: AuditSummary[]
  selected: Set<string>
  onToggle: (path: string) => void
}

export const ExperimentList = ({ audits, selected, onToggle }: ExperimentListProps) => (
  <div id="experiment-list" style={{ ...CardStyle, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
    <SectionTitle>Current Experiments</SectionTitle>
    <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '6px', marginTop: '8px', background: 'var(--bg)', borderRadius: 'var(--radius)', padding: '8px' }}>
      {audits.map(audit => (
        <ExperimentRow key={audit.path} audit={audit} checked={selected.has(audit.path)} onToggle={() => onToggle(audit.path)} />
      ))}
    </div>
    <SectionTitle>Audit Records</SectionTitle>
    <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '6px', marginTop: '8px', background: 'var(--bg)', borderRadius: 'var(--radius)', padding: '12px 8px 8px' }}>
      {audits.map(audit => (
        <ExperimentRow key={audit.path} audit={audit} checked={selected.has(audit.path)} onToggle={() => onToggle(audit.path)} />
      ))}
    </div>
  </div>
)
