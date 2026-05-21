import type { VerifyCategoryResult } from '../../types/ablation'

interface VerifyResultsProps {
  results: VerifyCategoryResult[]
}

const Pct = ({ value }: { value: number }) => <>{(value * 100).toFixed(0)}%</>

export const VerifyResults = ({ results }: VerifyResultsProps) => {
  if (results.length === 0) return null
  return (
    <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '16px' }}>
      <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text)', marginBottom: '8px' }}>
        Verify · before → after
      </div>
      <table style={{ width: '100%', fontSize: '11px', color: 'var(--text-dim)', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ color: 'var(--text-muted)', textAlign: 'left' }}>
            <th style={{ padding: '4px 6px' }}>category</th>
            <th style={{ padding: '4px 6px' }}>projection</th>
            <th style={{ padding: '4px 6px' }}>refusal rate</th>
          </tr>
        </thead>
        <tbody>
          {results.map(row => (
            <tr key={row.category} style={{ borderTop: '1px solid var(--border)' }}>
              <td style={{ padding: '4px 6px', color: 'var(--text)' }}>{row.category}</td>
              <td style={{ padding: '4px 6px' }}>
                {row.projection_before.toFixed(3)} → {row.projection_after.toFixed(3)}
              </td>
              <td style={{ padding: '4px 6px' }}>
                <Pct value={row.refusal_rate_before} /> → <Pct value={row.refusal_rate_after} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
