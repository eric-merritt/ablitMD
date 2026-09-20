import { FieldLabel } from './FieldLabel'
import { NumberInput } from './NumberInput'

export const Field = ({ label, value, onChange }: { label: string; value: number; onChange: (n: number) => void }) => (
  <label style={{ display: 'flex', flexDirection: 'column', alignItems: "center", gap: '4px' }}>
    <FieldLabel>{label}</FieldLabel>
    <NumberInput value={value} onChange={onChange} />
  </label>
)
