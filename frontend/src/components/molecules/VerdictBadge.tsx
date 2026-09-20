import type { AuditTrial } from '../../types/ablation'
import { JudgingDot } from '../atoms/JudgingDot'
import { JudgingLabel } from '../atoms/JudgingLabel'

export interface LiveTrial {
  index: number
  category: string
  prompt: string
  text: string
  streaming: boolean
  judging: boolean
  classification?: AuditTrial['classification']
  refused?: boolean
}

// Verdict badge: REFUSED / OK, or a "judging…" pulse while the judge is working.
export const VerdictBadge = ({ t }: { t: LiveTrial }) => {
  if (t.judging) {
    return (
      <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
        <JudgingDot />
        <JudgingLabel />
      </span>
    )
  }
  if (t.refused === undefined) return <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>·</span>
  return (
    <span style={{ fontWeight: 700, fontSize: '11px', letterSpacing: '0.06em', color: t.refused ? 'var(--danger)' : 'var(--ok)' }}>
      {t.refused ? 'REFUSED' : 'OK'}
    </span>
  )
}
