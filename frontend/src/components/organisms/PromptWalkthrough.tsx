import { useState, useEffect, useMemo } from 'react'
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
}

export const PromptWalkthrough = ({ initialRun, models, onComplete }: PromptWalkthroughProps) => {
  const { run, updatePromptResult, updateRunFields } = useRun(initialRun)
  const { modelLoading, generating, computing, ensureModelLoaded, generate, computeAllDirections } = useInference()

  const [promptIndex, setPromptIndex] = useState(0)
  const [response, setResponse] = useState<string | null>(null)
  const [genError, setGenError] = useState<string | null>(null)

  const currentStep = run ? run.sequence[run.current_sequence_index] : null

  const pendingPrompts: RunPrompt[] = useMemo(() => {
    if (!run || !currentStep) return []
    return run.prompts.filter(p => !p.model_results[currentStep.model]?.[currentStep.mode])
  }, [run, currentStep])

  const currentPrompt = pendingPrompts[promptIndex] ?? null
  const currentModel = models.find(m => m.modelId === currentStep?.model) ?? null

  useEffect(() => {
    if (!currentPrompt || !currentStep || !currentModel || !run) return

    setResponse(null)
    setGenError(null)

    const doGenerate = async () => {
      try {
        await ensureModelLoaded(currentModel.modelId, currentModel.apiModelId)
        const result = await generate({
          prompt_id: currentPrompt.prompt_id,
          prompt_text: currentPrompt.text,
          run_id: run.run_id,
          model_id: currentStep.model,
          mode: currentStep.mode,
        })
        setResponse(result.response)
      } catch (err: unknown) {
        setGenError(err instanceof Error ? err.message : 'Generation failed')
      }
    }

    doGenerate()
  }, [currentPrompt?.prompt_id, currentStep?.model, currentStep?.mode])

  const handleClassify = async (refused: boolean, mode: RefusalMode) => {
    if (!currentPrompt || !currentStep || !run) return

    const isLastPrompt = promptIndex >= pendingPrompts.length - 1
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

    if (!isLastPrompt) {
      setPromptIndex(prev => prev + 1)
      return
    }

    if (!isLastStep) {
      await updateRunFields({ current_sequence_index: run.current_sequence_index + 1 })
      setPromptIndex(0)
      return
    }

    const direction_results = await computeAllDirections(updatedRun)
    const finalRun = await patchRun(updatedRun.run_id, {
      direction_results,
      incomplete: false,
      completed_at: new Date().toISOString(),
    })
    onComplete(finalRun)
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

  return (
    <div style={{ maxWidth: '720px', margin: '0 auto', display: 'flex', flexDirection: 'column' }}>
      <RunProgress
        modelName={currentModel.name}
        mode={currentStep.mode}
        currentIndex={promptIndex}
        total={pendingPrompts.length}
      />
      <div style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
        {modelLoading && (
          <div style={{ padding: '10px', background: 'var(--accent-dim)', borderRadius: 'var(--radius)', color: 'var(--accent)', fontSize: '13px' }}>
            Loading {currentModel.name} into GPU…
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
  )
}
