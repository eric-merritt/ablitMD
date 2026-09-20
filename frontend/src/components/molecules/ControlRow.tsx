import { Field } from '../atoms/Field'
import { Button } from '../atoms/Button'

export const ControlRow = ({ running, nCategories, rounds, onNCategories, onRounds, onRun, onStop }: {
  running: boolean
  nCategories: number
  rounds: number
  onNCategories: (n: number) => void
  onRounds: (n: number) => void
  onRun: () => void
  onStop: () => void
}) => (
  <div style={{ display: 'flex', gap: '24px', alignItems: 'flex-end', justifyContent: 'center', marginTop: '12px', width: "100%" }}>
    <Field label="Test Categories" value={nCategories} onChange={onNCategories} />
    <span style={{ fontSize: '15px', color: 'var(--text-dim)', paddingBottom: '4px' }}>×</span>
    <Field label="Test Rounds" value={rounds} onChange={onRounds} />
    {running ? (
      <Button label="Stop" onClick={onStop} variant="danger" />
    ) : (
      <Button label="Run audit" onClick={onRun} />
    )}
  </div>
)
