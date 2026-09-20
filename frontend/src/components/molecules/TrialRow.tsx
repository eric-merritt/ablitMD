import { CardStyle } from '../atoms/CardStyle'
import { TrialCategory } from './TrialCategory'
import { EditButton } from './EditButton'
import { VerdictBadge, type LiveTrial } from './VerdictBadge'
import { TrialPrompt } from './TrialPrompt'
import { TrialText } from './TrialText'
import { Generating } from './Generating'

export const TrialRow = ({ t, onReclassify }: { t: LiveTrial; onReclassify: (index: number) => void }) => (
  <div style={{ ...CardStyle, padding: '12px 16px' }}>
    {/* Header: category left, classification cluster (pencil + label) right */}
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
      <TrialCategory id={t.category} />
      <span style={{ display: 'inline-flex', alignItems: 'center', gap: '8px' }}>
        {t.refused !== undefined && <EditButton onClick={() => onReclassify(t.index)} />}
        <VerdictBadge t={t} />
      </span>
    </div>

    {/* Body flush-left */}
    <TrialPrompt prompt={t.prompt} />
    {t.text ? <TrialText t={t} /> : t.streaming && <Generating />}
  </div>
)
