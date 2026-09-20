import { CardStyle } from '../atoms/CardStyle'
import { SectionTitle } from '../atoms/SectionTitle'
import { AgentBubble, type LiveTrial } from './AgentBubble'
import { StageLabel } from './StageLabel'

export const AgentFeed = ({ feed, stage, onReclassify }: { feed: LiveTrial[]; stage: string | null; onReclassify: (index: number) => void }) => (
  <div id="saved-audits-block" style={{ background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '16px', display: 'flex', flexDirection: 'column', gap: '8px', flex: 1, minHeight: 0 }}>
    <SectionTitle>Agent</SectionTitle>
    {stage && <StageLabel stage={stage} />}
    <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '8px', minHeight: 0 }}>
      {feed.map(t => (
        <AgentBubble key={t.index} t={t} onReclassify={onReclassify} />
      ))}
    </div>
  </div>
)
