import { useState } from 'react'
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

  const modelNames = Object.fromEntries(models.map(m => [m.modelId, m.name]))
  const walkthroughModels = models.map(m => ({ modelId: m.modelId, apiModelId: m.apiModelId, name: m.name }))

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
