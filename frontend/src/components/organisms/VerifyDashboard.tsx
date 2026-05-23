import { useEffect, useState } from 'react'
import { verifyAblation, verifyAblationClassic, bakeModel } from '../../api/ablation'
import type { VerifyPromptResult, VerifyCategoryResult } from '../../types/ablation'

interface VerifyDashboardProps {
  runId: string
  modelId: string
  genMode: string
  mode: 'ablitmd' | 'classic'
  classicFactor: number
  disclaimerAblate: boolean
  disclaimerFactor: number
  samplesPerCategory: number
  onBack: () => void
  onHome: () => void
}

const RefusalBadge = ({ refused }: { refused: boolean }) => (
  <span style={{
    display: 'inline-block', padding: '2px 8px', borderRadius: '4px', fontSize: '11px',
    fontWeight: 600, letterSpacing: '0.04em', textTransform: 'uppercase',
    background: refused ? '#7f1d1d' : '#064e3b', color: refused ? '#fca5a5' : '#6ee7b7',
  }}>
    { refused ? 'refused' : 'complied' }
  </span>
)

const Pct = ({ value }: { value: number }) => <>{ (value * 100).toFixed(0) }%</>

const ResponseBlock = ({ label, text, refused }: { label: string; text: string; refused: boolean }) => (
  <div style={{ flex: 1, minWidth: 0 }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
      <span style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>{ label }</span>
      <RefusalBadge refused={ refused } />
    </div>
    <div style={{
      background: 'var(--surface-3)', border: '1px solid var(--border)', borderRadius: 'var(--radius)',
      padding: '8px 10px', fontSize: '12px', color: 'var(--text-dim)', whiteSpace: 'pre-wrap',
      maxHeight: '180px', overflow: 'auto',
    }}>{ text || '(empty)' }</div>
  </div>
)

const PromptRow = ({ result }: { result: VerifyPromptResult }) => (
  <div style={{
    background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius)',
    padding: '12px', display: 'flex', flexDirection: 'column', gap: '10px',
  }}>
    <div style={{ display: 'flex', justifyContent: 'space-between', gap: '12px' }}>
      <span style={{ color: 'var(--accent)', fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.06em' }}>{ result.category }</span>
      <span style={{ color: 'var(--text-muted)', fontSize: '11px' }}>{ result.prompt_id }</span>
    </div>
    <div style={{ fontSize: '12px', color: 'var(--text)' }}>{ result.prompt_text }</div>
    <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
      <ResponseBlock label="before" text={ result.response_before } refused={ result.refused_before } />
      <ResponseBlock label="after" text={ result.response_after } refused={ result.refused_after } />
    </div>
  </div>
)

const ProgressHeader = ({ done, total, currentCategory }: { done: number; total: number; currentCategory: string }) => {
  const pct = total > 0 ? (done / total) * 100 : 0
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: 'var(--text-dim)' }}>
        <span>{ done } / { total } prompts · current: <code>{ currentCategory || '—' }</code></span>
        <span><Pct value={ total > 0 ? done / total : 0 } /></span>
      </div>
      <div style={{ height: '4px', background: 'var(--surface-3)', borderRadius: '2px', overflow: 'hidden' }}>
        <div style={{ height: '100%', background: 'var(--accent)', width: `${ pct }%`, transition: 'width 0.2s' }} />
      </div>
    </div>
  )
}

const CategorySummary = ({ rows }: { rows: VerifyCategoryResult[] }) => {
  if (rows.length === 0) return null
  return (
    <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '12px' }}>
      <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text)', marginBottom: '8px' }}>Per-category refusal rate</div>
      <table style={{ width: '100%', fontSize: '11px', color: 'var(--text-dim)', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ color: 'var(--text-muted)', textAlign: 'left' }}>
            <th style={{ padding: '4px 6px' }}>category</th>
            <th style={{ padding: '4px 6px' }}>refusal before → after</th>
            <th style={{ padding: '4px 6px' }}>projection before → after</th>
          </tr>
        </thead>
        <tbody>
          { rows.map(row => (
            <tr key={ row.category } style={{ borderTop: '1px solid var(--border)' }}>
              <td style={{ padding: '4px 6px', color: 'var(--text)' }}>{ row.category }</td>
              <td style={{ padding: '4px 6px' }}><Pct value={ row.refusal_rate_before } /> → <Pct value={ row.refusal_rate_after } /></td>
              <td style={{ padding: '4px 6px' }}>{ row.projection_before.toFixed(3) } → { row.projection_after.toFixed(3) }</td>
            </tr>
          )) }
        </tbody>
      </table>
    </div>
  )
}

export const VerifyDashboard = ({ runId, modelId, genMode, mode, classicFactor, disclaimerAblate, disclaimerFactor, samplesPerCategory, onBack, onHome }: VerifyDashboardProps) => {
  const [total, setTotal]                 = useState(0)
  const [done, setDone]                   = useState(0)
  const [currentCategory, setCurrentCategory] = useState('')
  const [prompts, setPrompts]             = useState<VerifyPromptResult[]>([])
  const [categories, setCategories]       = useState<VerifyCategoryResult[]>([])
  const [error, setError]                 = useState<string>()
  const [finished, setFinished]           = useState(false)
  const [baking, setBaking]               = useState(false)
  const [bakedPath, setBakedPath]         = useState<string>()

  const runBake = async () => {
    setBaking(true)
    try {
      const result = await bakeModel(
        runId,
        mode,
        mode === 'classic' ? classicFactor : undefined,
        mode === 'classic' ? disclaimerAblate : undefined,
        mode === 'classic' ? disclaimerFactor : undefined
      )
      setBakedPath(result.saved_to)
    } catch (err) {
      setError(`Bake failed: ${String((err as Error).message)}`)
    } finally {
      setBaking(false)
    }
  }

  useEffect(() => {
    const controller = new AbortController()
    let cancelled = false

    const stream = mode === 'classic'
      ? verifyAblationClassic(runId, { model_id: modelId, gen_mode: genMode, factor: classicFactor, disclaimer_ablate: disclaimerAblate, disclaimer_factor: disclaimerFactor, samples_per_category: samplesPerCategory }, onEvent, controller.signal)
      : verifyAblation(runId, { model_id: modelId, gen_mode: genMode, samples_per_category: samplesPerCategory }, onEvent, controller.signal)

    function onEvent(event: Parameters<typeof verifyAblation>[2] extends (e: infer E) => void ? E : never) {
      if (cancelled) return
      if (event.type === 'total') setTotal(event.prompts)
      else if (event.type === 'category_start') setCurrentCategory(event.category)
      else if (event.type === 'prompt') { setPrompts(prev => [...prev, event]); setDone(prev => prev + 1) }
      else if (event.type === 'category_result') setCategories(prev => [...prev, event])
    }

    stream
      .then(() => { if (!cancelled) setFinished(true) })
      .catch(err => { if (!cancelled && err.name !== 'AbortError') setError(String(err.message)) })

    return () => {
      cancelled = true
      controller.abort()
      fetch('/api/inference/reset', { method: 'POST' }).catch(() => {})
    }
  }, [runId, modelId, genMode, mode, classicFactor, samplesPerCategory])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{ flex: 1, overflow: 'auto' }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <span style={{ fontSize: '15px', fontWeight: 700, color: 'var(--text)' }}>
                Verify · { finished ? 'done' : 'running…' }
              </span>
              <span style={{
                padding: '2px 8px', borderRadius: '4px', fontSize: '11px', fontWeight: 600,
                background: mode === 'classic' ? '#1e40af' : 'var(--accent)',
                color: '#fff', textTransform: 'uppercase', letterSpacing: '0.04em',
              }}>
                { mode === 'classic' ? `classic ×${classicFactor.toFixed(2)}` : 'ablitMD' }
              </span>
            </div>
            <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
              <button onClick={ runBake } disabled={ !finished || baking }
                style={{
                  padding: '6px 14px', fontSize: '12px', cursor: !finished || baking ? 'not-allowed' : 'pointer',
                  background: 'var(--accent)', color: 'var(--bg)', border: 'none', borderRadius: 'var(--radius)',
                  fontWeight: 600, opacity: !finished || baking ? 0.5 : 1,
                }}>
                {baking ? 'Baking…' : 'Bake & Save'}
              </button>
              <span onClick={ onBack } style={{ cursor: 'pointer', color: 'var(--text-muted)', fontSize: '13px' }}>← Back</span>
              <span onClick={ onHome } style={{ cursor: 'pointer', color: 'var(--text-muted)', fontSize: '13px' }}>Home</span>
            </div>
          </div>

          { error && <div style={{ color: '#ef4444', fontSize: '12px' }}>{ error }</div> }
          { bakedPath && <div style={{ color: 'var(--accent)', fontSize: '12px' }}>Saved abliterated model to <code>{ bakedPath }</code></div> }

          <ProgressHeader done={ done } total={ total } currentCategory={ currentCategory } />

          <CategorySummary rows={ categories } />

          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            { prompts.slice().reverse().map(prompt => (
              <PromptRow key={ `${ prompt.category }__${ prompt.prompt_id }` } result={ prompt } />
            )) }
          </div>
        </div>
      </div>
    </div>
  )
}
