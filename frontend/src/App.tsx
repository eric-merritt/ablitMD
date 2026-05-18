import { useState, useMemo } from 'react'
import './App.css'
import { RunConfigPanel } from './components/organisms/RunConfigPanel'
import { PromptWalkthrough } from './components/organisms/PromptWalkthrough'
import { ClassifyReview } from './components/organisms/ClassifyReview'
import { ResultsGrid } from './components/organisms/ResultsGrid'
import { useModels } from './hooks/useModels'
import type { Run } from './types/run'

type Phase = 'config' | 'running' | 'review' | 'results'

const hasUnclassified = (run: Run): boolean =>
  run.prompts.some(prompt =>
    Object.values(prompt.model_results || {}).some(modeMap =>
      Object.values(modeMap || {}).some(result => result && result.refusal_mode === undefined)
    )
  )

const App = () => {
  const { models } = useModels()
  const [phase, setPhase] = useState<Phase>('config')
  const [activeRun, setActiveRun] = useState<Run | null>(null)

  const modelNames = useMemo(
    () => Object.fromEntries(models.map(model => [model.modelId, model.name])),
    [models]
  )
  const walkthroughModels = useMemo(
    () => models.map(model => ({ modelId: model.modelId, apiModelId: model.apiModelId, name: model.name })),
    [models]
  )

  const handleRunStart       = (run: Run) => { setActiveRun(run); setPhase('running') }
  const handleRunOpen        = (run: Run) => {
    setActiveRun(run)
    if (run.incomplete) { setPhase('running'); return }
    setPhase(hasUnclassified(run) ? 'review' : 'results')
  }
  const handleReadyForReview = (run: Run) => { setActiveRun(run); setPhase('review') }
  const handleRunComplete    = (run: Run) => { setActiveRun(run); setPhase('results') }

  return (
    <>
      { phase === 'config' && (
        <RunConfigPanel models={ models } onRunStart={ handleRunStart } onRunOpen={ handleRunOpen } />
      ) }
      { phase === 'running' && activeRun && (
        <PromptWalkthrough
          initialRun={ activeRun }
          models={ walkthroughModels }
          onReadyForReview={ handleReadyForReview }
          onBack={ () => setPhase('config') }
          onHome={ () => setPhase('config') }
        />
      ) }
      { phase === 'review' && activeRun && (
        <ClassifyReview
          run={ activeRun }
          modelNames={ modelNames }
          onComplete={ handleRunComplete }
          onBack={ () => setPhase('running') }
          onHome={ () => setPhase('config') }
        />
      ) }
      { phase === 'results' && activeRun && (
        <ResultsGrid
          run={ activeRun }
          modelNames={ modelNames }
          onBack={ () => setPhase('review') }
          onHome={ () => setPhase('config') }
        />
      ) }
    </>
  )
}

export default App
