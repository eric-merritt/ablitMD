import { useState } from 'react'
import type { CSSProperties } from 'react'
import { ModeRadioGroup } from '../molecules/ModeRadioGroup'
import { ModelCheckboxList } from '../molecules/ModelCheckboxList'
import { createRun, fetchRuns, fetchRun } from '../../api/runs'
import { CATEGORIES } from '../../types/categories'
import type { LLM } from '../../types/model'
import type { Run, RunSummary, RunMode } from '../../types/run'

interface RunConfigPanelProps {
  models: LLM[]
  onRunStart: (run: Run) => void
  onRunOpen: (run: Run) => void
}

const PanelStyle: CSSProperties = {
  maxWidth: '520px',
  margin: '48px auto',
  padding: '0 24px',
  display: 'flex',
  flexDirection: 'column',
  gap: '28px',
}

const btnBase: CSSProperties = {
  padding: '9px 20px',
  borderRadius: 'var(--radius)',
  border: 'none',
  fontWeight: 600,
}

const PrimaryBtn = (disabled: boolean): CSSProperties => ({
  ...btnBase,
  background: disabled ? '#333' : 'var(--accent)',
  color: disabled ? 'var(--text-muted)' : '#fff',
  cursor: disabled ? 'not-allowed' : 'pointer',
})

const SecondaryBtn: CSSProperties = {
  ...btnBase,
  background: 'var(--surface)',
  color: 'var(--text-dim)',
  border: '1px solid var(--border)',
  cursor: 'pointer',
}

export const RunConfigPanel = ({ models, onRunStart, onRunOpen }: RunConfigPanelProps) => {
  const [mode, setMode] = useState('non_thinking')
  const [selectedModels, setSelectedModels] = useState<Set<string>>(new Set())
  const [selectedCategories, setSelectedCategories] = useState<Set<string>>(
    new Set(CATEGORIES.map(cat => cat.id))
  )
  const [existingRuns, setExistingRuns] = useState<RunSummary[] | null>(null)
  const [loadingRuns, setLoadingRuns] = useState(false)
  const [starting, setStarting] = useState(false)

  const canStart = selectedModels.size > 0 && selectedCategories.size > 0

  const handleModelToggle = (modelId: string, checked: boolean) =>
    setSelectedModels(prev => {
      const next = new Set(prev)
      checked ? next.add(modelId) : next.delete(modelId)
      return next
    })

  const handleCategoryToggle = (categoryId: string, checked: boolean) =>
    setSelectedCategories(prev => {
      const next = new Set(prev)
      checked ? next.add(categoryId) : next.delete(categoryId)
      return next
    })

  const handleGroupToggle = (groupId: string, checked: boolean) => {
    const groupCatIds = CATEGORIES.filter(cat => cat.group === groupId).map(cat => cat.id)
    setSelectedCategories(prev => {
      const next = new Set(prev)
      groupCatIds.forEach(id => checked ? next.add(id) : next.delete(id))
      return next
    })
  }

  const handleStart = async () => {
    setStarting(true)
    try {
      const run = await createRun({
        models: [...selectedModels],
        mode_selection: mode as RunMode,
        prompt_scope: { categories: [...selectedCategories] },
      })
      onRunStart(run)
    } finally {
      setStarting(false)
    }
  }

  const handleOpenExisting = async () => {
    setLoadingRuns(true)
    try {
      const runs = await fetchRuns()
      setExistingRuns(runs)
    } finally {
      setLoadingRuns(false)
    }
  }

  const handleSelectRun = async (summary: RunSummary) => {
    const run = await fetchRun(summary.run_id)
    onRunOpen(run)
  }

  return (
    <div style={PanelStyle}>
      <ModeRadioGroup selected={mode} onChange={setMode} />
      <ModelCheckboxList
        models={models}
        selectedModels={selectedModels}
        selectedCategories={selectedCategories}
        onModelToggle={handleModelToggle}
        onCategoryToggle={handleCategoryToggle}
        onGroupToggle={handleGroupToggle}
      />
      <div style={{ display: 'flex', gap: '12px' }}>
        <button
          style={PrimaryBtn(!canStart || starting)}
          disabled={!canStart || starting}
          onClick={handleStart}
        >
          {starting ? 'Starting…' : 'Start Run'}
        </button>
        <button style={SecondaryBtn} onClick={handleOpenExisting} disabled={loadingRuns}>
          {loadingRuns ? 'Loading…' : 'Open Existing Run'}
        </button>
      </div>
      {existingRuns && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
          <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '6px' }}>
            Previous Runs
          </div>
          {existingRuns.length === 0 && (
            <div style={{ color: 'var(--text-muted)' }}>No runs yet.</div>
          )}
          {existingRuns.map(run => (
            <div
              key={run.run_id}
              onClick={() => handleSelectRun(run)}
              style={{ padding: '10px 12px', background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', cursor: 'pointer', display: 'flex', justifyContent: 'space-between' }}
            >
              <span style={{ color: 'var(--text-dim)' }}>
                {new Date(run.started_at).toLocaleString()} — {run.prompt_count} prompts
              </span>
              <span style={{ color: run.incomplete ? '#eab308' : '#10b981', fontSize: '12px' }}>
                {run.incomplete ? 'Incomplete' : 'Complete'}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
