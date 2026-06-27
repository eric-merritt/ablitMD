import { useState, useEffect, useMemo, useRef } from 'react'
import { RunProgress } from '../molecules/RunProgress'
import { PromptCard } from '../molecules/PromptCard'
import { ResponsePanel } from '../molecules/ResponsePanel'
import { useInference } from '../../hooks/useInference'
import { useRun } from '../../hooks/useRun'
import type { Run, RunPrompt } from '../../types/run'

interface WalkthroughModel {
  modelId: string
  apiModelId: string
  name: string
}

interface PromptWalkthroughProps {
  initialRun: Run
  models: WalkthroughModel[]
  onReadyForReview: (run: Run) => void
  onBack: () => void
  onHome: () => void
}

const ModelLoadingBar = ({ modelName }: { modelName: string }) => {
  const [elapsed, setElapsed] = useState(0)
  useEffect(() => {
    const tick = setInterval(() => setElapsed(seconds => seconds + 1), 1000)
    return () => clearInterval(tick)
  }, [])
  const mins = Math.floor(elapsed / 60)
  const secs = String(elapsed % 60).padStart(2, '0')
  const timeStr = mins > 0 ? `${mins}m ${secs}s` : `${elapsed}s`
  return (
    <div style={{ padding: '12px 14px', background: 'var(--accent-dim)', border: '1px solid var(--accent)44', borderRadius: 'var(--radius)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
        <span style={{ color: 'var(--accent)', fontSize: '13px' }}>Loading { modelName } into GPU…</span>
        <span style={{ color: 'var(--text-muted)', fontSize: '11px', fontVariantNumeric: 'tabular-nums' }}>{ timeStr }</span>
      </div>
      <div style={{ height: '3px', background: 'var(--surface-3)', borderRadius: '2px', overflow: 'hidden' }}>
        <div style={{
          height: '100%',
          width: '40%',
          background: 'var(--accent)',
          borderRadius: '2px',
          animation: 'inferenceSlide 1.6s ease-in-out infinite',
        }} />
      </div>
    </div>
  )
}

const HomeIcon = () => (
  <svg width="13" height="13" viewBox="0 0 14 14" fill="none" style={{ display: 'inline-block', verticalAlign: '-2px', marginRight: '5px' }}>
    <path d="M1 7L7 1L13 7V13H9.5V9H4.5V13H1V7Z" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round" fill="none" />
  </svg>
)

const NavBar = ({ onBack, onHome, disabled }: { onBack: () => void; onHome: () => void; disabled: boolean }) => (
  <div>
    <hr style={{ border: 'none', borderTop: '1px solid var(--border)', margin: 0 }} />
    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '10px 32px' }}>
      <span
        onClick={ disabled ? undefined : onBack }
        style={{ cursor: disabled ? 'default' : 'pointer', color: disabled ? 'var(--text-muted)' : 'var(--text)', fontSize: '19px', userSelect: 'none' }}
      >← Back</span>
      <span onClick={ onHome } style={{ cursor: 'pointer', color: 'var(--text)', fontSize: '19px', userSelect: 'none' }}>
        <HomeIcon />Home
      </span>
    </div>
  </div>
)

export const PromptWalkthrough = ({ initialRun, models, onReadyForReview, onBack, onHome }: PromptWalkthroughProps) => {
  const { run, updatePromptResult, updateRunFields } = useRun(initialRun)
  const { generating, ensureModelLoaded, generate } = useInference()

  const [modelReady, setModelReady] = useState(false)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [stepPosition, setStepPosition] = useState(0)
  const [responses, setResponses] = useState<Record<string, string>>({})
  const [genErrors, setGenErrors] = useState<Record<string, string>>({})
  const advancing = useRef(false)

  const currentStep = run ? run.sequence[run.current_sequence_index] : null

  const pendingPrompts: RunPrompt[] = useMemo(() => {
    if (!run || !currentStep) return []
    return run.prompts.filter(prompt => !prompt.model_results[currentStep.model]?.[currentStep.mode])
  }, [run, currentStep])

  const currentPrompt = pendingPrompts[0] ?? null
  const currentModel = models.find(model => model.modelId === currentStep?.model) ?? null
  const response = currentPrompt ? responses[currentPrompt.prompt_id] ?? null : null
  const genError = currentPrompt ? genErrors[currentPrompt.prompt_id] ?? null : null

  // Effect 0 — resumed onto an already-finished step (every prompt has a result): skip the
  // model load and generation entirely, and advance to the next step — or hand off to review
  // if it was the last. Without this the model would be loaded into GPU just to find no work.
  useEffect(() => {
    if (!run || !currentStep) return
    if (pendingPrompts.length > 0) return
    if (advancing.current) return
    advancing.current = true

    const isLastStep = run.current_sequence_index >= run.sequence.length - 1
    if (isLastStep) {
      onReadyForReview(run)
      return
    }

    updateRunFields({ current_sequence_index: run.current_sequence_index + 1 })
      .finally(() => { advancing.current = false })
    setStepPosition(0)
    setResponses({})
    setGenErrors({})
  }, [currentStep?.model, currentStep?.mode, pendingPrompts.length, run])

  // Effect 1 — load model whenever the active model changes, but only if this step has pending work
  useEffect(() => {
    if (!currentModel) return
    if (pendingPrompts.length === 0) return
    setModelReady(false)
    setLoadError(null)
    ensureModelLoaded(currentModel.modelId, currentModel.apiModelId)
      .then(() => setModelReady(true))
      .catch(err => setLoadError(err instanceof Error ? err.message : 'Failed to load model'))
  }, [currentStep?.model])

  // Effect 2 — stream generate, save result (no classification), auto-advance
  useEffect(() => {
    if (!modelReady || !currentPrompt || !currentStep || !currentModel || !run) return
    if (responses[currentPrompt.prompt_id] !== undefined) return

    const promptId = currentPrompt.prompt_id
    const controller = new AbortController()
    setResponses(prev => ({ ...prev, [promptId]: '' }))

    const runAutoStep = async () => {
      try {
        const result = await generate({
          prompt_id: promptId,
          prompt_text: currentPrompt.text,
          run_id: run.run_id,
          model_id: currentStep.model,
          mode: currentStep.mode,
        }, (token) => {
          setResponses(prev => ({ ...prev, [promptId]: (prev[promptId] ?? '') + token }))
        }, controller.signal)

        setResponses(prev => ({ ...prev, [promptId]: result.response }))
        await persistAndAdvance(result.response, result.hidden_states_key, promptId)
      } catch (err: unknown) {
        if ((err as Error)?.name === 'AbortError') return
        setGenErrors(prev => ({ ...prev, [promptId]: err instanceof Error ? err.message : 'Generation failed' }))
      }
    }

    const persistAndAdvance = async (responseText: string, hiddenStatesKey: string, promptId: string) => {
      if (advancing.current) return
      advancing.current = true
      try {
        const partialResult = {
          response: responseText,
          hidden_states_key: hiddenStatesKey,
        }
        const updatedRun = await updatePromptResult(promptId, currentStep.model, currentStep.mode, partialResult)
        if (!updatedRun) return

        const stillPending = updatedRun.prompts.some(prompt => !prompt.model_results[currentStep.model]?.[currentStep.mode])
        const isLastStep = updatedRun.current_sequence_index >= updatedRun.sequence.length - 1

        if (stillPending) {
          setStepPosition(prev => prev + 1)
          return
        }

        if (!isLastStep) {
          await updateRunFields({ current_sequence_index: updatedRun.current_sequence_index + 1 })
          setStepPosition(0)
          setResponses({})
          setGenErrors({})
          return
        }

        onReadyForReview(updatedRun)
      } finally {
        advancing.current = false
      }
    }

    runAutoStep()
    return () => controller.abort()
  }, [modelReady, currentPrompt?.prompt_id, currentStep?.model, currentStep?.mode])

  if (!run || !currentStep || !currentModel) {
    return <div style={{ padding: '48px', color: 'var(--text-muted)' }}>Loading run…</div>
  }

  if (!modelReady && pendingPrompts.length > 0) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', padding: '48px', maxWidth: '480px', width: '100%', margin: '0 auto' }}>
          { loadError
            ? <div style={{ color: '#ef4444', fontSize: '13px' }}>{ loadError }</div>
            : <ModelLoadingBar modelName={ currentModel.name } />
          }
        </div>
        <NavBar onBack={ onBack } onHome={ onHome } disabled={ false } />
      </div>
    )
  }

  if (!currentPrompt) {
    return <div style={{ padding: '48px', color: 'var(--text-muted)' }}>Preparing review…</div>
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{ flex: 1, overflow: 'auto', maxWidth: '720px', width: '100%', margin: '0 auto' }}>
        <RunProgress
          modelName={ currentModel.name }
          mode={ currentStep.mode }
          currentIndex={ stepPosition }
          total={ pendingPrompts.length + stepPosition }
        />
        <div style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <PromptCard prompt={ currentPrompt } />
          <ResponsePanel generating={ generating } response={ response } error={ genError } />
          <div style={{ padding: '10px 12px', background: 'var(--surface-2)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', color: 'var(--text-muted)', fontSize: '12px', textAlign: 'center' }}>
            Autorunning — classification deferred to review screen
          </div>
        </div>
      </div>
      <NavBar onBack={ onBack } onHome={ onHome } disabled={ generating } />
    </div>
  )
}
