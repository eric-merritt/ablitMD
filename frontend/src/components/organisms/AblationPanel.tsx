import { useEffect, useState } from 'react'
import { DivergenceChart } from '../molecules/DivergenceChart'
import { LayerSplitControls } from '../molecules/LayerSplitControls'
import { RecipePanel } from '../molecules/RecipePanel'
import { VerifyResults } from '../molecules/VerifyResults'
import { getDivergence, buildRecipe, applyAblation, clearAblation, verifyAblation, bakeModel } from '../../api/ablation'
import type { ModeDivergence, SlimRecipe, VerifyCategoryResult } from '../../types/ablation'

interface AblationPanelProps {
  runId: string
}

const pickFirstModeMap = (payload: Record<string, unknown>) => {
  // payload = { computed_at, [model]: { [gen_mode]: { hard?, redirect? } } }
  for (const [key, value] of Object.entries(payload)) {
    if (key !== 'computed_at' && value && typeof value === 'object') {
      const genModes = value as Record<string, Record<string, ModeDivergence>>
      const firstGenMode = Object.values(genModes)[0]
      if (firstGenMode) return firstGenMode
    }
  }
  return null
}

export const AblationPanel = ({ runId }: AblationPanelProps) => {
  const [hard, setHard]         = useState<ModeDivergence>()
  const [redirect, setRedirect] = useState<ModeDivergence>()
  const [onset, setOnset]       = useState(38)
  const [split, setSplit]       = useState(50)
  const [factorA, setFactorA]   = useState(0.15)
  const [factorB, setFactorB]   = useState(0.15)
  const [recipe, setRecipe]     = useState<SlimRecipe>()
  const [status, setStatus]     = useState<'idle' | 'loading' | 'building' | 'applied'>('loading')
  const [error, setError]       = useState<string>()
  const [verifyResults, setVerifyResults] = useState<VerifyCategoryResult[]>([])
  const [verifying, setVerifying] = useState(false)
  const [bakedPath, setBakedPath] = useState<string>()
  const [baking, setBaking]       = useState(false)

  useEffect(() => {
    getDivergence(runId)
      .then(payload => {
        const modeMap = pickFirstModeMap(payload as Record<string, unknown>)
        if (!modeMap) { setError('No divergence data for this run'); setStatus('idle'); return }
        setHard(modeMap.hard)
        setRedirect(modeMap.redirect)
        const reference = modeMap.hard ?? modeMap.redirect
        if (reference) {
          setOnset(reference.suggested_onset)
          setSplit(reference.suggested_divergence)
        }
        setStatus('idle')
      })
      .catch(err => { setError(String(err.message)); setStatus('idle') })
  }, [runId])

  const rebuildRecipe = () => {
    setStatus('building')
    setError(undefined)
    buildRecipe(runId, { onset, split, factorA, factorB })
      .then(next => { setRecipe(next); setStatus('idle') })
      .catch(err => { setError(String(err.message)); setStatus('idle') })
  }

  const runVerify = () => {
    if (!recipe) return
    setVerifying(true)
    setVerifyResults([])
    setError(undefined)
    verifyAblation(runId, { model_id: recipe.model_id, gen_mode: recipe.gen_mode },
      result => setVerifyResults(prev => [...prev, result]))
      .catch(err => setError(String(err.message)))
      .finally(() => setVerifying(false))
  }

  const runBake = () => {
    setBaking(true)
    setError(undefined)
    bakeModel(runId)
      .then(result => setBakedPath(result.saved_to))
      .catch(err => setError(String(err.message)))
      .finally(() => setBaking(false))
  }

  const lastLayer = (hard ?? redirect)?.clumping.length
    ? ((hard ?? redirect)!.clumping.length - 1)
    : split

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <div style={{ fontSize: '15px', fontWeight: 700, color: 'var(--text)' }}>
        Abliteration Pipeline
      </div>

      {error && <div style={{ color: '#ef4444', fontSize: '12px' }}>{error}</div>}
      {status === 'loading' && <div style={{ color: 'var(--text-muted)', fontSize: '12px' }}>Loading divergence…</div>}

      {(hard || redirect) && (
        <>
          <DivergenceChart hard={hard} redirect={redirect} onset={onset} split={split} />
          <div style={{ display: 'flex', gap: '20px', alignItems: 'flex-end', flexWrap: 'wrap' }}>
            <LayerSplitControls onset={onset} split={split} lastLayer={lastLayer}
              onChange={next => { setOnset(next.onset); setSplit(next.split) }} />
            <button onClick={rebuildRecipe} disabled={status === 'building'}
              style={{ padding: '6px 14px', cursor: 'pointer' }}>
              {status === 'building' ? 'Building…' : 'Build recipe'}
            </button>
          </div>
        </>
      )}

      {recipe && (
        <>
          <RecipePanel recipe={recipe} factorA={factorA} factorB={factorB}
            onFactorChange={next => { setFactorA(next.factorA); setFactorB(next.factorB) }} />
          <div style={{ display: 'flex', gap: '12px' }}>
            <button onClick={() => applyAblation(runId).then(() => setStatus('applied')).catch(err => setError(String(err.message)))}
              style={{ padding: '6px 14px', cursor: 'pointer' }}>
              Apply hooks
            </button>
            <button onClick={() => clearAblation(runId).then(() => setStatus('idle')).catch(err => setError(String(err.message)))}
              style={{ padding: '6px 14px', cursor: 'pointer' }}>
              Clear hooks
            </button>
            <button onClick={runVerify} disabled={verifying}
              style={{ padding: '6px 14px', cursor: 'pointer' }}>
              {verifying ? 'Verifying…' : 'Verify'}
            </button>
            <button onClick={runBake} disabled={baking}
              style={{ padding: '6px 14px', cursor: 'pointer' }}>
              {baking ? 'Baking…' : 'Bake & save model'}
            </button>
            {status === 'applied' && <span style={{ color: 'var(--accent)', fontSize: '12px', alignSelf: 'center' }}>hooks active</span>}
          </div>
          <VerifyResults results={verifyResults} />
          {bakedPath && (
            <div style={{ color: 'var(--accent)', fontSize: '12px' }}>
              Saved abliterated model to <code>{bakedPath}</code>
            </div>
          )}
        </>
      )}
    </div>
  )
}
