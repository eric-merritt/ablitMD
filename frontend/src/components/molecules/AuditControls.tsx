import { CardStyle } from '../atoms/CardStyle'
import { SectionTitle } from '../atoms/SectionTitle'
import { ErrorText } from '../atoms/ErrorText'
import { ControlRow } from './ControlRow'

export const AuditControls = ({ running, nCategories, rounds, error, onNCategories, onRounds, onRun, onStop }: {
  running: boolean
  nCategories: number
  rounds: number
  error: string | null
  onNCategories: (n: number) => void
  onRounds: (n: number) => void
  onRun: () => void
  onStop: () => void
}) => (
  <div id="audit-controls" style={{ ...CardStyle, display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center' }}>
    <SectionTitle>Adversarial Audit</SectionTitle>
    <ControlRow
      running={running}
      nCategories={nCategories}
      rounds={rounds}
      onNCategories={onNCategories}
      onRounds={onRounds}
      onRun={onRun}
      onStop={onStop}
    />
    {error && <ErrorText message={error} />}
  </div>
)
