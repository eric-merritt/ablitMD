import { TrialCategory } from './TrialCategory'
import { EditButton } from './EditButton'
import { VerdictBadge, type LiveTrial } from './VerdictBadge'
import { TrialPrompt } from './TrialPrompt'
import { TrialText } from './TrialText'
import { Generating } from './Generating'

export const AgentBubble = ({ t, onReclassify }: { t: LiveTrial; onReclassify: (index: number) => void }) => (
  <div style={{ background: 'var(--surface-3)', borderRadius: '12px', padding: '10px 14px', maxWidth: '95%' }}>
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '6px' }}>
      <TrialCategory id={t.category} />
      <span style={{ display: 'inline-flex', alignItems: 'center', gap: '8px' }}>
        {t.refused !== undefined && <EditButton onClick={() => onReclassify(t.index)} />}
        <VerdictBadge t={t} />
      </span>
    </div>
    <TrialPrompt prompt={t.prompt} />
    {t.text ? <TrialText t={t} /> : t.streaming && <Generating />}
  </div>
)
