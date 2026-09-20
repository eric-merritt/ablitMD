import type { ReactNode } from 'react'

export const FieldLabel = ({ children }: { children: ReactNode }) => (
  <span style={{ fontSize: '14px', color: 'var(--text-dim)' }}>{children}</span>
)
