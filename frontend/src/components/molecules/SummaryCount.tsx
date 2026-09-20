export const SummaryCount = ({ nRefused, nTotal }: { nRefused: number; nTotal: number }) => (
  <span style={{ fontSize: '13px', fontWeight: 600, color: nRefused > 0 ? 'var(--danger)' : 'var(--ok)' }}>
    {nRefused} refused / {nTotal}
  </span>
)
