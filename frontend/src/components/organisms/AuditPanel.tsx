import { useEffect, useRef, useState } from 'react'
import { runAudit, listAudits } from '../../api/ablation'
import type { AuditRecord, AuditSummary, AuditTrial } from '../../types/ablation'

interface AuditPanelProps {
  run: { run_id: string }
}

const CATEGORY_NAMES: Record<string, string> = {}
// Populated lazily below to avoid importing the whole CATEGORIES list here.
import { CATEGORIES } from '../../types/categories'
for (const cat of CATEGORIES) CATEGORY_NAMES[cat.id] = cat.name

const labelOf = (id: string): string => CATEGORY_NAMES[id] ?? id

const fmtTime = (iso: string | null | undefined): string => {
  if (!iso) return '—'
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString()
}

// One row in the live feed. `streaming` means the model is still writing it;
// `judging` means generation finished and we're waiting on the judge's label.
interface LiveTrial {
  index: number
  category: string
  prompt: string
  text: string
  streaming: boolean
  judging: boolean
  classification?: AuditTrial['classification']
  refused?: boolean
}

const PENCIL = (
  <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 20h9" />
    <path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z" />
  </svg>
)

// Verdict badge: REFUSED / OK, or a "judging…" pulse while the judge is working.
const VerdictBadge = ({ t }: { t: LiveTrial }) => {
  if (t.judging) {
    return (
      <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', color: 'var(--text-muted)', fontSize: '11px' }}>
        <span style={{ width: '7px', height: '7px', borderRadius: '50%', background: 'var(--accent)', animation: 'auditPulse 1s ease-in-out infinite' }} />
        judging…
      </span>
    )
  }
  if (t.refused === undefined) return <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>·</span>
  return (
    <span style={{ fontWeight: 700, fontSize: '11px', letterSpacing: '0.04em', color: t.refused ? '#ef4444' : '#22c55e' }}>
      {t.refused ? 'REFUSED' : 'OK'}
    </span>
  )
}

// Pencil button — dark gray, rounded-square border on hover, lightens to match.
const EditButton = ({ onClick }: { onClick: () => void }) => (
  <button
    onClick={onClick}
    title="Reclassify"
    className="audit-edit"
    style={{
      display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
      width: '24px', height: '24px', padding: 0, margin: 0,
      background: 'transparent', color: '#6e7681', border: '1px solid transparent',
      borderRadius: '6px', cursor: 'pointer', transition: 'color .12s, border-color .12s',
    }}
  >
    {PENCIL}
  </button>
)

const TrialRow = ({ t, onReclassify }: { t: LiveTrial; onReclassify: (index: number) => void }) => (
  <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '12px 16px' }}>
    {/* Header: category left, classification cluster (pencil + label) top-right */}
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
      <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>{labelOf(t.category)}</span>
      <span style={{ display: 'inline-flex', alignItems: 'center', gap: '8px' }}>
        {t.refused !== undefined && <EditButton onClick={() => onReclassify(t.index)} />}
        <VerdictBadge t={t} />
      </span>
    </div>

    {/* Body flush-left */}
    <p style={{ fontSize: '12px', color: 'var(--text-dim)', margin: '0 0 6px' }}>“{t.prompt}”</p>
    {t.text ? (
      <p style={{ margin: 0, fontSize: '13px', color: 'var(--text)', whiteSpace: 'pre-wrap', lineHeight: 1.55 }}>
        {t.text}
        {t.streaming && <span style={{ display: 'inline-block', width: '2px', height: '1em', background: 'var(--accent)', marginLeft: '1px', verticalAlign: 'text-bottom', animation: 'auditBlink 1s step-end infinite' }} />}
      </p>
    ) : (
      t.streaming && <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>generating…</div>
    )}
  </div>
)

// Confirm-before-reclassify modal.
const ReclassifyModal = ({ onConfirm, onCancel }: { onConfirm: () => void; onCancel: () => void }) => (
  <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,.6)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 50 }} onClick={e => e.target === e.currentTarget && onCancel()}>
    <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: '10px', padding: '22px', width: '340px', boxShadow: '0 12px 40px rgba(0,0,0,.5)' }}>
      <h2 style={{ fontSize: '14px', margin: '0 0 8px', color: 'var(--text)' }}>Reclassify this response?</h2>
      <p style={{ fontSize: '12px', color: 'var(--text-dim)', lineHeight: 1.5, margin: '0 0 16px' }}>
        The judge will re-read the model's answer and update its verdict in the record.
      </p>
      <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
        <button onClick={onCancel} style={{ background: 'transparent', color: 'var(--text-dim)', border: '1px solid var(--border)' }}>Cancel</button>
        <button onClick={onConfirm}>Yes, reclassify</button>
      </div>
    </div>
  </div>
)

export const AuditPanel = ({ run }: AuditPanelProps) => {
  const [nCategories, setNCategories] = useState(5)
  const [rounds, setRounds] = useState(3)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [latest, setLatest] = useState<AuditRecord | null>(null)
  const [audits, setAudits] = useState<AuditSummary[]>([])
  const [feed, setFeed] = useState<LiveTrial[]>([])
  const [pendingReclassify, setPendingReclassify] = useState<number | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const refreshList = () => {
    listAudits(run.run_id).then(setAudits).catch(() => setAudits([]))
  }

  useEffect(refreshList, [run.run_id])

  useEffect(() => () => abortRef.current?.abort(), [])

  const handleRun = async () => {
    setRunning(true)
    setError(null)
    setLatest(null)
    setFeed([])
    const controller = new AbortController()
    abortRef.current = controller

    try {
      await runAudit(run.run_id, (event) => {
        if (controller.signal.aborted) return
        switch (event.type) {
          case 'trial_start':
            setFeed(prev => [
              { index: event.index, category: event.category, prompt: event.prompt, text: '', streaming: true, judging: false },
              ...prev,
            ])
            break
          case 'token':
            // Append to the newest (top) row — that's the one generating.
            setFeed(prev => prev.map((t, i) => (i === 0 ? { ...t, text: t.text + event.text } : t)))
            break
          case 'trial_done':
            setFeed(prev => prev.map(t =>
              t.index === event.index
                ? { ...t, streaming: false, judging: false, classification: event.classification, refused: event.refused, text: event.response }
                : t,
            ))
            break
          case 'audit_done':
            setLatest(event.record)
            refreshList()
            break
          case 'error':
            setError(event.message)
            break
        }
      }, nCategories, rounds, controller.signal)
    } catch (err) {
      if (!(err instanceof DOMException && err.name === 'AbortError'))
        setError(err instanceof Error ? err.message : String(err))
    } finally {
      setRunning(false)
    }
  }

  const handleStop = () => abortRef.current?.abort()

  // "Yes" definitively flips the card's verdict to the opposite side. The judge's
  // classification label is left as-is — it's the reason, not a boolean.
  const confirmReclassify = (index: number) => {
    setPendingReclassify(null)
    setFeed(prev => prev.map(t => t.index === index && t.refused !== undefined ? { ...t, refused: !t.refused } : t))
    setLatest(prev => prev ? { ...prev, trials: prev.trials.map((t, i) => (i === index ? { ...t, refused: !t.refused } : t)) } : prev)
  }

  return (
    <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap' }}>
      {/* Controls + live feed */}
      <div style={{ flex: '1 1 360px', minWidth: '300px' }}>
        <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '10px' }}>
          Adversarial Audit
        </div>
        <p style={{ fontSize: '12px', color: 'var(--text-dim)', margin: '0 0 14px', lineHeight: 1.5 }}>
          Runs the ablated model against fresh random prompts from a random subset of categories,
          and judges each new response as refused / not-refused with the 9B judge. Watch it generate
          live — click the pencil to reclassify a response.
        </p>

        <div style={{ display: 'flex', gap: '16px', alignItems: 'flex-end', marginBottom: '14px' }}>
          <label style={{ display: 'flex', flexDirection: 'column', gap: '4px', fontSize: '12px', color: 'var(--text-dim)' }}>
            Categories / round
            <input
              type="number" min={1} max={50} value={nCategories}
              onChange={e => setNCategories(Math.max(1, Number(e.target.value) || 1))}
              style={{ width: '90px' }}
            />
          </label>
          <label style={{ display: 'flex', flexDirection: 'column', gap: '4px', fontSize: '12px', color: 'var(--text-dim)' }}>
            Rounds
            <input
              type="number" min={1} max={50} value={rounds}
              onChange={e => setRounds(Math.max(1, Number(e.target.value) || 1))}
              style={{ width: '90px' }}
            />
          </label>
          {running ? (
            <button onClick={handleStop}>Stop</button>
          ) : (
            <button onClick={handleRun}>Run audit</button>
          )}
        </div>

        {error && <div style={{ fontSize: '12px', color: '#ef4444', marginBottom: '10px' }}>{error}</div>}

        {/* Live feed — newest trial on top, streams in as it happens */}
        {feed.length > 0 && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginTop: '8px' }}>
            {feed.map(t => (
              <TrialRow key={t.index} t={t} onReclassify={setPendingReclassify} />
            ))}
          </div>
        )}

        {/* Final summary once the audit completes */}
        {latest && (
          <div style={{ marginTop: '14px', fontSize: '12px', color: 'var(--text-dim)' }}>
            Audit complete · {latest.trials.length} trials ·{' '}
            <span style={{ color: latest.trials.some(t => t.refused) ? '#ef4444' : '#22c55e', fontWeight: 600 }}>
              {latest.trials.filter(t => t.refused).length} refused
            </span>
          </div>
        )}
      </div>

      {/* Saved experiments */}
      <div style={{ flex: '1 1 240px', minWidth: '220px' }}>
        <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '10px' }}>
          Saved audits ({audits.length})
        </div>
        {audits.length === 0 && (
          <div style={{ fontSize: '12px', color: 'var(--text-dim)' }}>No audits yet for this run.</div>
        )}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
          {audits.map(audit => (
            <div key={audit.path} style={{ fontSize: '12px', color: 'var(--text-dim)', display: 'flex', justifyContent: 'space-between', gap: '8px' }}>
              <span>{fmtTime(audit.created_at)}</span>
              <span>{audit.n_trials} trials · {audit.n_refused} refused</span>
            </div>
          ))}
        </div>
      </div>

      {pendingReclassify !== null && (
        <ReclassifyModal onConfirm={() => confirmReclassify(pendingReclassify)} onCancel={() => setPendingReclassify(null)} />
      )}

      <style>{`
        @keyframes auditBlink { 50% { opacity: 0; } }
        @keyframes auditPulse { 0%,100% { opacity: .3; transform: scale(.85); } 50% { opacity: 1; transform: scale(1); } }
        .audit-edit:hover { color: #c9d1d9 !important; border-color: #c9d1d9 !important; }
      `}</style>
    </div>
  )
}
