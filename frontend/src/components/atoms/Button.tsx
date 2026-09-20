import type { CSSProperties } from 'react'

export interface ButtonProps {
  label: string
  onClick: () => void
  variant?: 'primary' | 'danger' | 'ghost'
}

const variants: Record<string, CSSProperties> = {
  primary: {
    background: 'var(--accent)',
    color: 'var(--text)',
    border: '1px solid var(--accent)',
  },
  danger: {
    background: 'transparent',
    color: 'var(--danger)',
    border: '1px solid var(--border-2)',
  },
  ghost: {
    background: 'transparent',
    color: 'var(--text-dim)',
    border: '1px solid var(--border)',
  },
}

export const Button = ({ label, onClick, variant = 'primary' }: ButtonProps) => (
  <button
    onClick={onClick}
    style={{
      ...variants[variant],
      borderRadius: '6px', padding: '8px 18px', fontSize: '15px', cursor: 'pointer',
    }}
  >
    {label}
  </button>
)
