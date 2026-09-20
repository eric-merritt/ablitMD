import { useState, useMemo } from 'react'
import './App.css'
import { RunConfigPanel } from './components/organisms/RunConfigPanel'
import { RunModeChoice } from './components/molecules/RunModeChoice'
import { PromptWalkthrough } from './components/organisms/PromptWalkthrough'
import { AutoClassifyReview } from './components/organisms/AutoClassifyReview'
import { ClassifyReview } from './components/organisms/ClassifyReview'
import { ResultsGrid } from './components/organisms/ResultsGrid'
import { VerifyDashboard } from './components/organisms/VerifyDashboard'
import { AuditPanel } from './components/organisms/AuditPanel'
import { useModels } from './hooks/useModels'
import type { Run } from './types/run'

type Phase = 'config' | 'running' | 'auto-review' | 'review' | 'choose-mode' | 'results' | 'verify' | 'audit'

type AblationMode = 'ablitmd' | 'classic'
interface VerifyContext { genMode: string; samplesPerCategory: number; mode: AblationMode; classicFactor: number; disclaimerAblate: boolean; disclaimerFactor: number }

const App = () => {
  const { models } = useModels()
  const [phase, setPhase] = useState<Phase>('config')
  const [activeRun, setActiveRun] = useState<Run | null>(null)
  const [verifyContext, setVerifyContext] = useState<VerifyContext | null>(null)

  const handleVerifyStart = (genMode: string, mode: AblationMode, classicFactor: number, disclaimerAblate: boolean, disclaimerFactor: number) => {
    setVerifyContext({ genMode, samplesPerCategory: 2, mode, classicFactor, disclaimerAblate, disclaimerFactor })
    setPhase('verify')
  }

  const handleAuditStart = () => setPhase('audit')

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
    // a run with computed direction results has its charts ready — open straight to them
    if (run.direction_results && Object.keys(run.direction_results).length > 0) {
      setPhase('results')
      return
    }
    if (run.incomplete) { setPhase('running'); return }
    // Completed run: let the user pick how to abliterate it.
    setPhase('choose-mode')
  }
  const handleReadyForReview = (run: Run) => { setActiveRun(run); setPhase('auto-review') }
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
      { phase === 'auto-review' && activeRun && (
        <AutoClassifyReview
          run={ activeRun }
          modelNames={ modelNames }
          onComplete={ handleRunComplete }
          onManualReview={ () => setPhase('review') }
          onBack={ () => setPhase('running') }
          onHome={ () => setPhase('config') }
        />
      ) }
      { phase === 'review' && activeRun && (
        <ClassifyReview
          run={ activeRun }
          modelNames={ modelNames }
          onComplete={ handleRunComplete }
          onBack={ () => setPhase('auto-review') }
          onHome={ () => setPhase('config') }
        />
      ) }
      { phase === 'choose-mode' && activeRun && (
        <RunModeChoice onManual={() => setPhase('results')} onAudit={handleAuditStart} />
      ) }
      { phase === 'results' && activeRun && (
        <ResultsGrid
          run={ activeRun }
          modelNames={ modelNames }
          models={ walkthroughModels }
          onVerify={ handleVerifyStart }
          onAudit={ handleAuditStart }
          onBack={ () => setPhase('review') }
          onHome={ () => setPhase('config') }
        />
      ) }
      { phase === 'audit' && activeRun && (
					<>
          	<AuditPanel run={ activeRun } />
            <hr style={{ border: 'none', borderTop: '1px solid var(--border)', margin: 0 }} />
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 32px' }}>
              <span onClick={() => setPhase('results')} style={{ color: 'var(--text)', fontSize: '19px', cursor: 'pointer', userSelect: 'none' }}>← Back</span>
              <span onClick={() => setPhase('config')} style={{ color: 'var(--text)', fontSize: '19px', cursor: 'pointer', userSelect: 'none' }}>Home</span>
						</div>
					</>
      )}
      { phase === 'verify' && activeRun && verifyContext && (
        <VerifyDashboard
          runId={ activeRun.run_id }
          genMode={ verifyContext.genMode }
          mode={ verifyContext.mode }
          classicFactor={ verifyContext.classicFactor }
          disclaimerAblate={ verifyContext.disclaimerAblate }
          disclaimerFactor={ verifyContext.disclaimerFactor }
          samplesPerCategory={ verifyContext.samplesPerCategory }
          onBack={ () => setPhase('results') }
          onHome={ () => setPhase('config') }
        />
      ) }
    </>
  )
}

export default App
