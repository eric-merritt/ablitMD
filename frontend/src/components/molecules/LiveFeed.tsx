import { TrialRow, type LiveTrial } from './TrialRow'

export const LiveFeed = ({ feed, onReclassify }: { feed: LiveTrial[]; onReclassify: (index: number) => void }) => (
  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
    {feed.map(t => (
      <TrialRow key={t.index} t={t} onReclassify={onReclassify} />
    ))}
  </div>
)
