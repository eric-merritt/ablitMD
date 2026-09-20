import type { ReactNode } from 'react'

export const SectionTitle = ({ children }: { children: ReactNode }) => (
  <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
    {children}
  </div>
)
