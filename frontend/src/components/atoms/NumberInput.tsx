export const NumberInput = ({ value, onChange }: { value: number; onChange: (n: number) => void }) => (
  <input
    type="number" min={1} max={50} value={value}
    onChange={e => onChange(Math.max(1, Number(e.target.value) || 1))}
    style={{ width: '90px', fontSize: '15px' }}
  />
)
