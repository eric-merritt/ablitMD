import { useState, useMemo } from 'react'
import './App.css'
import { RunConfigPanel } from './components/organisms/RunConfigPanel'
import { PromptWalkthrough } from './components/organisms/PromptWalkthrough'
import { ResultsGrid } from './components/organisms/ResultsGrid'
import { useModels } from './hooks/useModels'
import type { Run } from './types/run'

type Phase = 'config' | 'running' | 'results'

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

  const handleRunStart = (run: Run) => { setActiveRun(run); setPhase('running') }
  const handleRunOpen  = (run: Run) => { setActiveRun(run); setPhase(run.incomplete ? 'running' : 'results') }
  const handleRunComplete = (run: Run) => { setActiveRun(run); setPhase('results') }

  return (
    <>
      {phase === 'config' && (
        <RunConfigPanel models={models} onRunStart={handleRunStart} onRunOpen={handleRunOpen} />
      )}
      {phase === 'running' && activeRun && (
        <PromptWalkthrough initialRun={activeRun} models={walkthroughModels} onComplete={handleRunComplete} />
      )}
      {phase === 'results' && activeRun && (
        <ResultsGrid run={activeRun} modelNames={modelNames} />
      )}
    </>
  )
}

export default App
