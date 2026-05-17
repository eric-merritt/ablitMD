import { useState, useEffect, useMemo, useRef } from 'react'
import { RunProgress } from '../molecules/RunProgress'
import { PromptCard } from '../molecules/PromptCard'
import { ResponsePanel } from '../molecules/ResponsePanel'
import { RefusalClassifier } from '../molecules/RefusalClassifier'
import { useInference } from '../../hooks/useInference'
import { useRun } from '../../hooks/useRun'
import { patchRun } from '../../api/runs'
import type { Run, RunPrompt, RefusalMode } from '../../types/run'

interface WalkthroughModel {
  modelId: string
  apiModelId: string
  name: string
}

interface PromptWalkthroughProps {
  initialRun: Run
  models: WalkthroughModel[]
  onComplete: (run: Run) => void
  onBack: () => void
  onHome: () => void
}

const HomeIcon = () => (
  <svg width="13" height="13" viewBox="0 0 14 14" fill="none" style={{ display: 'inline-block', verticalAlign: '-2px', marginRight: '5px' }}>
    <path d="M1 7L7 1L13 7V13H9.5V9H4.5V13H1V7Z" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round" fill="none" />
  </svg>
)

export const PromptWalkthrough = ({ initialRun, models, onComplete, onBack, onHome }: PromptWalkthroughProps) => {
  const { run, updatePromptResult, updateRunFields } = useRun(initialRun)
  const { modelLoading, generating, computing, ensureModelLoaded, generate, computeAllDirections } = useInference()

  const [promptIndex, setPromptIndex] = useState(0)
  const [skippedIds, setSkippedIds] = useState<Set<string>>(new Set())
  const [responses, setResponses] = useState<Record<string, string>>({})
  const [genErrors, setGenErrors] = useState<Record<string, string>>({})
  const [backHovered, setBackHovered] = useState(false)
  const [skipHovered, setSkipHovered] = useState(false)
  const [homeHovered, setHomeHovered] = useState(false)
  const classifying = useRef(false)

  const currentStep = run ? run.sequence[run.current_sequence_index] : null

  const sortedPendingPrompts: RunPrompt[] = useMemo(() => {
    if (!run || !currentStep) return []
    const pending = run.prompts.filter(p => !p.model_results[currentStep.model]?.[currentStep.mode])
    return [
      ...pending.filter(p => !skippedIds.has(p.prompt_id)),
      ...pending.filter(p => skippedIds.has(p.prompt_id)),
    ]
  }, [run, currentStep, skippedIds])

  const currentPrompt = sortedPendingPrompts[promptIndex] ?? null
  const currentModel = models.find(m => m.modelId === currentStep?.model) ?? null
  const response = currentPrompt ? responses[currentPrompt.prompt_id] ?? null : null
  const genError = currentPrompt ? genErrors[currentPrompt.prompt_id] ?? null : null
  const isOnSkipped = currentPrompt ? skippedIds.has(currentPrompt.prompt_id) : false
  const skippedRemaining = sortedPendingPrompts.filter(p => skippedIds.has(p.prompt_id)).length

  useEffect(() => {
    if (!currentPrompt || !currentStep || !currentModel || !run) return
    if (responses[currentPrompt.prompt_id]) return

    const promptId = currentPrompt.prompt_id
    const doGenerate = async () => {
      try {
        await ensureModelLoaded(currentModel.modelId, currentModel.apiModelId)
        const result = await generate({
          prompt_id: promptId,
          prompt_text: currentPrompt.text,
          run_id: run.run_id,
          model_id: currentStep.model,
          mode: currentStep.mode,
        })
        setResponses(prev => ({ ...prev, [promptId]: result.response }))
      } catch (err: unknown) {
        setGenErrors(prev => ({ ...prev, [promptId]: err instanceof Error ? err.message : 'Generation failed' }))
      }
    }
    doGenerate()
  }, [currentPrompt?.prompt_id, currentStep?.model, currentStep?.mode])

  const handleSkip = () => {
    if (!currentPrompt || generating || modelLoading || isOnSkipped) return
    setSkippedIds(prev => new Set([...prev, currentPrompt.prompt_id]))
  }

  const handleBack = () => {
    if (generating || modelLoading) return
    if (promptIndex === 0) onBack()
    else setPromptIndex(prev => prev - 1)
  }

  const handleClassify = async (refused: boolean, mode: RefusalMode) => {
    if (classifying.current) return
    if (!currentPrompt || !currentStep || !run) return
    classifying.current = true
    try {
      const isLastPrompt = promptIndex >= sortedPendingPrompts.length - 1
      const isLastStep = run.current_sequence_index >= run.sequence.length - 1

      const result = {
        response: response ?? '',
        refused,
        refusal_mode: mode,
        classified_at: new Date().toISOString(),
        hidden_states_key: `${currentPrompt.prompt_id}__${currentStep.model.replace('/', '__')}__${currentStep.mode}`,
      }

      const updatedRun = await updatePromptResult(
        currentPrompt.prompt_id, currentStep.model, currentStep.mode, result
      )
      if (!updatedRun) return

      setSkippedIds(prev => { const next = new Set(prev); next.delete(currentPrompt.prompt_id); return next })

      if (!isLastPrompt) return

      if (!isLastStep) {
        await updateRunFields({ current_sequence_index: updatedRun.current_sequence_index + 1 })
        setPromptIndex(0)
        setSkippedIds(new Set())
        setResponses({})
        setGenErrors({})
        return
      }

      const direction_results = await computeAllDirections(updatedRun)
      const finalRun = await patchRun(updatedRun.run_id, {
        direction_results,
        incomplete: false,
        completed_at: new Date().toISOString(),
      })
      onComplete(finalRun)
    } finally {
      classifying.current = false
    }
  }

  if (!run || !currentStep || !currentModel) {
    return <div style={{ padding: '48px', color: 'var(--text-muted)' }}>Loading run…</div>
  }

  if (!currentPrompt) {
    return (
      <div style={{ padding: '48px', color: 'var(--text-muted)' }}>
        {computing ? 'Computing directions…' : 'All prompts classified.'}
      </div>
    )
  }

  const navDisabled = generating || modelLoading

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{ flex: 1, overflow: 'auto', maxWidth: '720px', width: '100%', margin: '0 auto' }}>
        <RunProgress
          modelName={currentModel.name}
          mode={currentStep.mode}
          currentIndex={promptIndex}
          total={sortedPendingPrompts.length}
        />
        <div style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {modelLoading && (
            <div style={{ padding: '10px', background: 'var(--accent-dim)', borderRadius: 'var(--radius)', color: 'var(--accent)', fontSize: '13px' }}>
              Loading {currentModel.name} into GPU…
            </div>
          )}
          {isOnSkipped && (
            <div style={{ padding: '8px 12px', background: 'rgba(234,179,8,0.1)', border: '1px solid rgba(234,179,8,0.3)', borderRadius: 'var(--radius)', color: '#eab308', fontSize: '12px' }}>
              Revisiting skipped prompt — {skippedRemaining} remaining
            </div>
          )}
          <PromptCard prompt={currentPrompt} />
          <ResponsePanel generating={generating || modelLoading} response={response} error={genError} />
          <RefusalClassifier
            disabled={generating || modelLoading || !response}
            onClassify={handleClassify}
          />
        </div>
      </div>

      <div>
        <hr style={{ border: 'none', borderTop: '1px solid var(--border)', margin: 0 }} />
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 32px' }}>
          <span
            onClick={handleBack}
            onMouseEnter={() => setBackHovered(true)}
            onMouseLeave={() => setBackHovered(false)}
            style={{
              color: navDisabled ? 'var(--text-muted)' : backHovered ? 'var(--accent)' : 'var(--text)',
              fontSize: '19px',
              cursor: navDisabled ? 'default' : 'pointer',
              userSelect: 'none',
              transition: 'color 0.15s',
            }}
          >
            ←{' '}
            <span style={{ textDecoration: !navDisabled && backHovered ? 'underline' : 'none' }}>
              Back
            </span>
          </span>

          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '2px' }}>
            {skippedRemaining > 0 && !isOnSkipped && (
              <span style={{ fontSize: '11px', color: '#eab308' }}>
                {skippedRemaining} skipped
              </span>
            )}
            <span
              onClick={onHome}
              onMouseEnter={() => setHomeHovered(true)}
              onMouseLeave={() => setHomeHovered(false)}
              style={{
                color: homeHovered ? 'var(--accent)' : 'var(--text)',
                fontSize: '19px',
                cursor: 'pointer',
                userSelect: 'none',
                transition: 'color 0.15s',
              }}
            >
              <HomeIcon />
              <span style={{ textDecoration: homeHovered ? 'underline' : 'none' }}>Home</span>
            </span>
          </div>

          <span
            onClick={handleSkip}
            onMouseEnter={() => setSkipHovered(true)}
            onMouseLeave={() => setSkipHovered(false)}
            style={{
              color: navDisabled || isOnSkipped ? 'var(--text-muted)' : skipHovered ? 'var(--accent)' : 'var(--text)',
              fontSize: '19px',
              cursor: navDisabled || isOnSkipped ? 'default' : 'pointer',
              userSelect: 'none',
              transition: 'color 0.15s',
            }}
          >
            <span style={{ textDecoration: !navDisabled && !isOnSkipped && skipHovered ? 'underline' : 'none' }}>
              Skip
            </span>
            {' '}→
          </span>
        </div>
      </div>
    </div>
  )
}
