import { useEffect, useRef, useState } from 'react'
import { runAuditFull, listAudits } from '../../api/ablation'
import { inferenceLoad, inferenceStatus } from '../../api/inference'
import type { AuditRecord, AuditSummary, AuditTrial } from '../../types/ablation'
import { AuditControls } from '../molecules/AuditControls'
import { ExperimentList } from '../molecules/ExperimentList'
import { OverlapWorkspace } from '../molecules/OverlapWorkspace'
import { AgentFeed } from '../molecules/AgentFeed'
import { ReclassifyModal } from '../molecules/ReclassifyModal'
import { PanelStyles } from '../atoms/PanelStyles'

interface AuditPanelProps {
  run: { run_id: string; sequence?: { model: string; mode: string }[] }
}

// One row in the live feed. `streaming` means the model is still writing it;
// `judging` means generation finished and we're waiting on the judge's label.
export interface LiveTrial {
  index: number
  category: string
  prompt: string
  text: string
  streaming: boolean
  judging: boolean
  classification?: AuditTrial['classification']
  refused?: boolean
}

// --- panel -------------------------------------------------------------------

export const AuditPanel = ({ run }: AuditPanelProps) => {
  const [nCategories, setNCategories] = useState(5)
  const [rounds, setRounds] = useState(3)
  const [running, setRunning] = useState(false)
  const [stage, setStage] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [latest, setLatest] = useState<AuditRecord | null>(null)
  const [audits, setAudits] = useState<AuditSummary[]>([])
  const [feed, setFeed] = useState<LiveTrial[]>([])
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [pendingReclassify, setPendingReclassify] = useState<number | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const toggleExperiment = (path: string) => {
    setSelected(prev => {
      const next = new Set(prev)
      if (next.has(path)) next.delete(path)
      else next.add(path)
      return next
    })
  }

  const refreshList = () => {
    listAudits(run.run_id).then(setAudits).catch(() => setAudits([]))
  }

  useEffect(refreshList, [run.run_id])

  // Pre-load the model so it's resident in memory before the user clicks Run.
  // This avoids OOM from cold-start loading during the audit itself.
  useEffect(() => {
    const step = run.sequence?.[0]
    if (!step) return

    const load = async () => {
      try {
        const status = await inferenceStatus()
        if (status.loaded_model !== step.model) {
          console.log(`[audit] pre-loading model: ${step.model}`)
          await inferenceLoad({ model_id: step.model, api_model_id: step.model })
          console.log(`[audit] model loaded: ${step.model}`)
        }
      } catch (e) {
        console.warn('[audit] pre-load failed — audit will load on demand:', e)
      }
    }
    load()
  }, [run.run_id, run.sequence])

  useEffect(() => () => abortRef.current?.abort(), [])

  const handleRun = async () => {
    setRunning(true)
    setError(null)
    setStage(null)
    setLatest(null)
    setFeed([])
    const controller = new AbortController()
    abortRef.current = controller

    // Show direction arrows immediately when the button is clicked.
    setSelected(prev => new Set([...prev, '__recipe__']))

    try {
      await runAuditFull(run.run_id, (event) => {
        if (controller.signal.aborted) return
        switch (event.type) {
          case 'stage':
            // Self-contained flow progress: building_recipe / loading_model / abliterating / auditing.
            setStage(event.stage)
            break
          case 'trial_start':
            setFeed(prev => [
              { index: event.index, category: event.category, prompt: event.prompt, text: '', streaming: true, judging: false },
              ...prev,
            ])
            break
          case 'token':
            // Append to the newest (top) row — that's the one generating.
            setFeed(prev => prev.map((t, i) => (i === 0 ? { ...t, text: t.text + event.text } : t)))
            break
          case 'trial_done':
            setFeed(prev => prev.map(t =>
              t.index === event.index
                ? { ...t, streaming: false, judging: false, classification: event.classification, refused: event.refused, text: event.response }
                : t,
            ))
            break
          case 'audit_done':
            setLatest(event.record)
            refreshList()
            // Add the new audit to selected so its category data shows in overlap too.
            if (event.record.path) {
              setSelected(prev => new Set([...prev, event.record.path]))
            }
            break
          case 'error':
            setError(event.message)
            break
        }
      }, nCategories, rounds, controller.signal)
    } catch (err) {
      if (!(err instanceof DOMException && err.name === 'AbortError'))
        setError(err instanceof Error ? err.message : String(err))
    } finally {
      setRunning(false)
    }
  }

  const handleStop = () => abortRef.current?.abort()

  // "Yes" definitively flips the card's verdict to the opposite side. The judge's
  // classification label is left as-is — it's the reason, not a boolean.
  const confirmReclassify = (index: number) => {
    setPendingReclassify(null)
    setFeed(prev => prev.map(t => t.index === index && t.refused !== undefined ? { ...t, refused: !t.refused } : t))
    setLatest(prev => prev ? { ...prev, trials: prev.trials.map((t, i) => (i === index ? { ...t, refused: !t.refused } : t)) } : prev)
  }

  return (
    <div id="audit-panel">
      <ExperimentList audits={audits} selected={selected} onToggle={toggleExperiment} />
      <AuditControls
        running={running}
        nCategories={nCategories}
        rounds={rounds}
        error={error}
        onNCategories={setNCategories}
        onRounds={setRounds}
        onRun={handleRun}
        onStop={handleStop}
      />
      <OverlapWorkspace run={run} selected={selected} />
      <AgentFeed feed={feed} stage={stage} onReclassify={setPendingReclassify} />
      {pendingReclassify !== null && (
        <ReclassifyModal onConfirm={() => confirmReclassify(pendingReclassify)} onCancel={() => setPendingReclassify(null)} />
      )}
      <PanelStyles />
    </div>
  )
}
