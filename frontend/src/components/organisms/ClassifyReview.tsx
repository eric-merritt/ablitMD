import { useState, useMemo } from 'react'
import { useInference } from '../../hooks/useInference'
import { patchRun, patchRunPrompt } from '../../api/runs'
import type { Run, RunPrompt, RefusalMode } from '../../types/run'

interface ClassifyReviewProps {
  run: Run
  modelNames: Record<string, string>
  onComplete: (run: Run) => void
  onBack: () => void
  onHome: () => void
}

type Selection = RefusalMode | 'none' | null

interface ReviewRowData {
  prompt: RunPrompt
  modelId: string
  mode: string
  response: string
  hiddenStatesKey: string
  existingSelection: Selection
}

const HomeIcon = () => (
  <svg width="13" height="13" viewBox="0 0 14 14" fill="none" style={{ display: 'inline-block', verticalAlign: '-2px', marginRight: '5px' }}>
    <path d="M1 7L7 1L13 7V13H9.5V9H4.5V13H1V7Z" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round" fill="none" />
  </svg>
)

const existingSelectionFor = (refused?: boolean, refusal_mode?: RefusalMode): Selection => {
  if (refusal_mode === undefined) return null
  if (refusal_mode === 'none' || refused === false) return 'none'
  return refusal_mode
}

const collectRows = (run: Run): ReviewRowData[] => {
  const rows: ReviewRowData[] = []
  for (const prompt of run.prompts) {
    for (const [modelId, modeMap] of Object.entries(prompt.model_results || {})) {
      for (const [mode, result] of Object.entries(modeMap || {})) {
        if (!result || !result.response) continue
        rows.push({
          prompt,
          modelId,
          mode,
          response: result.response,
          hiddenStatesKey: result.hidden_states_key ?? '',
          existingSelection: existingSelectionFor( result.refused, result.refusal_mode ),
        })
      }
    }
  }
  return rows
}

const SelectButton = ({ active, color, label, onClick }: {
  active: boolean
  color: string
  label: string
  onClick: () => void
}) => (
  <button
    onClick={ onClick }
    style={{
      padding: '6px 10px',
      borderRadius: 'var(--radius)',
      border: `1px solid ${ active ? color : 'var(--border)' }`,
      background: active ? `${ color }15` : 'transparent',
      color: active ? color : 'var(--text)',
      fontSize: '12px',
      fontWeight: active ? 600 : 400,
      cursor: 'pointer',
      textAlign: 'left',
    }}
  >{ label }</button>
)

const RowClassifier = ({ selection, onChange }: {
  selection: Selection
  onChange: (value: Selection) => void
}) => (
  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', minWidth: '140px' }}>
    <SelectButton active={ selection === 'none' }       color="#16a34a" label="Not Refused" onClick={ () => onChange( 'none' ) } />
    <SelectButton active={ selection === 'hard' }       color="#ef4444" label="Hard"        onClick={ () => onChange( 'hard' ) } />
    <SelectButton active={ selection === 'redirect' }   color="#eab308" label="Redirect"    onClick={ () => onChange( 'redirect' ) } />
    <SelectButton active={ selection === 'disclaimer' } color="#3b82f6" label="Disclaimer"  onClick={ () => onChange( 'disclaimer' ) } />
  </div>
)

const SavedBadge = () => (
  <span style={{ fontSize: '10px', color: 'var(--text-muted)', padding: '1px 6px', border: '1px solid var(--border)', borderRadius: '10px' }}>saved</span>
)

const PromptCell = ({ row, preClassified }: { row: ReviewRowData; preClassified: boolean }) => (
  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
    <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', alignItems: 'center' }}>
      <span style={{ fontSize: '10px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>{ row.prompt.category }</span>
      <span style={{ fontSize: '10px', color: row.prompt.type === 'harmful' ? '#ef4444' : '#16a34a' }}>{ row.prompt.type }</span>
      { preClassified && <SavedBadge /> }
    </div>
    <div style={{ fontSize: '13px', color: 'var(--text)', whiteSpace: 'pre-wrap', lineHeight: 1.5 }}>{ row.prompt.text }</div>
  </div>
)

const ResponseCell = ({ response }: { response: string }) => (
  <div style={{ fontSize: '13px', color: 'var(--text)', whiteSpace: 'pre-wrap', lineHeight: 1.5, maxHeight: '320px', overflow: 'auto', padding: '8px 10px', background: 'var(--surface-2)', borderRadius: 'var(--radius)', border: '1px solid var(--border)' }}>{ response }</div>
)

const ReviewRow = ({ row, selection, onChange }: {
  row: ReviewRowData
  selection: Selection
  onChange: (value: Selection) => void
}) => (
  <div style={{
    display: 'grid',
    gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1.4fr) auto',
    gap: '14px',
    padding: '14px 16px',
    borderTop: '1px solid var(--border)',
  }}>
    <PromptCell row={ row } preClassified={ row.existingSelection !== null } />
    <ResponseCell response={ row.response } />
    <RowClassifier selection={ selection } onChange={ onChange } />
  </div>
)

const SubmitBar = ({ classifiedCount, total, submitting, onSubmit, onBack, onHome }: {
  classifiedCount: number
  total: number
  submitting: boolean
  onSubmit: () => void
  onBack: () => void
  onHome: () => void
}) => {
  const ready = classifiedCount === total && !submitting
  return (
    <div>
      <hr style={{ border: 'none', borderTop: '1px solid var(--border)', margin: 0 }} />
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 32px' }}>
        <span onClick={ submitting ? undefined : onBack } style={{ cursor: submitting ? 'default' : 'pointer', color: submitting ? 'var(--text-muted)' : 'var(--text)', fontSize: '19px', userSelect: 'none' }}>← Back</span>
        <span style={{ fontSize: '13px', color: 'var(--text-muted)', fontVariantNumeric: 'tabular-nums' }}>
          { classifiedCount } / { total } classified
        </span>
        <span
          onClick={ ready ? onSubmit : undefined }
          style={{ cursor: ready ? 'pointer' : 'default', color: ready ? 'var(--accent)' : 'var(--text-muted)', fontSize: '19px', fontWeight: 600, userSelect: 'none' }}
        >{ submitting ? 'Submitting…' : 'Submit →' }</span>
        <span onClick={ onHome } style={{ cursor: 'pointer', color: 'var(--text)', fontSize: '19px', userSelect: 'none' }}><HomeIcon />Home</span>
      </div>
    </div>
  )
}

const rowKey = (row: ReviewRowData): string => `${ row.prompt.prompt_id }__${ row.modelId }__${ row.mode }`

const seedSelections = (rows: ReviewRowData[]): Record<string, Selection> =>
  Object.fromEntries( rows.map( row => [rowKey( row ), row.existingSelection] ) )

export const ClassifyReview = ({ run, modelNames, onComplete, onBack, onHome }: ClassifyReviewProps) => {
  const { computeAllDirections } = useInference()

  const rows = useMemo( () => collectRows( run ), [run] )
  const [selections, setSelections] = useState<Record<string, Selection>>( () => seedSelections( rows ) )
  const [submitting, setSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState<string | null>(null)

  const classifiedCount = Object.values( selections ).filter( value => value !== null && value !== undefined ).length

  const handleSubmit = async () => {
    if (submitting) return
    setSubmitting(true)
    setSubmitError(null)
    try {
      for (const row of rows) {
        const selection = selections[rowKey( row )]
        if (!selection) continue
        if (selection === row.existingSelection) continue
        const refused = selection !== 'none'
        const refusal_mode: RefusalMode = refused ? selection as RefusalMode : 'none'
        await patchRunPrompt(run.run_id, row.prompt.prompt_id, row.modelId, row.mode, {
          response: row.response,
          hidden_states_key: row.hiddenStatesKey,
          refused,
          refusal_mode,
          classified_at: new Date().toISOString(),
        })
      }

      const direction_results = await computeAllDirections( run )
      const finalRun = await patchRun(run.run_id, {
        direction_results,
        incomplete: false,
        completed_at: new Date().toISOString(),
      })
      onComplete(finalRun)
    } catch (err: unknown) {
      setSubmitError(err instanceof Error ? err.message : 'Submit failed')
      setSubmitting(false)
    }
  }

  if (rows.length === 0) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
        <div style={{ flex: 1, padding: '48px', color: 'var(--text-muted)', textAlign: 'center' }}>
          No responses to review yet.
        </div>
        <SubmitBar classifiedCount={ 0 } total={ 0 } submitting={ false } onSubmit={ handleSubmit } onBack={ onBack } onHome={ onHome } />
      </div>
    )
  }

  const headerLabel = `${ modelNames[rows[0].modelId] ?? rows[0].modelId } · ${ rows[0].mode }`
  const newCount = rows.filter( row => row.existingSelection === null ).length

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{ flex: 1, overflow: 'auto' }}>
        <div style={{ maxWidth: '1280px', margin: '0 auto', padding: '20px 16px 8px' }}>
          <div style={{ fontSize: '13px', color: 'var(--text-muted)', marginBottom: '6px' }}>Review &amp; classify</div>
          <div style={{ fontSize: '18px', fontWeight: 600, color: 'var(--text)' }}>{ headerLabel }</div>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '4px' }}>
            { rows.length } total · { newCount } new · { rows.length - newCount } previously saved
          </div>
          { submitError && <div style={{ marginTop: '12px', color: '#ef4444', fontSize: '13px' }}>{ submitError }</div> }
        </div>
        <div style={{ maxWidth: '1280px', margin: '0 auto' }}>
          { rows.map( row => (
            <ReviewRow
              key={ rowKey( row ) }
              row={ row }
              selection={ selections[rowKey( row )] ?? null }
              onChange={ (value) => setSelections( prev => ({ ...prev, [rowKey( row )]: value }) ) }
            />
          ) ) }
        </div>
      </div>
      <SubmitBar
        classifiedCount={ classifiedCount }
        total={ rows.length }
        submitting={ submitting }
        onSubmit={ handleSubmit }
        onBack={ onBack }
        onHome={ onHome }
      />
    </div>
  )
}
