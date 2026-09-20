import { Cursor } from '../atoms/Cursor'
import type { LiveTrial } from './VerdictBadge'

export const TrialText = ({ t }: { t: LiveTrial }) => (
  <p style={{ margin: 0, fontSize: '13px', color: 'var(--text)', whiteSpace: 'pre-wrap', lineHeight: 1.55 }}>
    {t.text}
    {t.streaming && <Cursor />}
  </p>
)
