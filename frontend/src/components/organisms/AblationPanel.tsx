import { useEffect, useState } from 'react'
import { DivergenceChart } from '../molecules/DivergenceChart'
import { LayerSplitControls } from '../molecules/LayerSplitControls'
import { RecipePanel } from '../molecules/RecipePanel'
import { VerifyResults } from '../molecules/VerifyResults'
import { getDivergence, buildRecipe, applyAblation, clearAblation, verifyAblation, bakeModel } from '../../api/ablation'
import { inferenceStatus, inferenceLoad } from '../../api/inference'
import type { ModeDivergence, SlimRecipe, VerifyCategoryResult } from '../../types/ablation'

interface AblationPanelProps {
  runId: string
  models: { modelId: string; apiModelId: string; name: string }[]
}

type ModelLoadState = 'idle' | 'loading' | 'ready' | 'error'

const GatedButton = ({ disabled, tooltip, onClick, children }: {
  disabled: boolean
  tooltip: string
  onClick: () => void
  children: React.ReactNode
}) => {
  const [hovered, setHovered] = useState(false)
  return (
    <span style={{ position: 'relative', display: 'inline-block' }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}>
      <button onClick={onClick} disabled={disabled}
        style={{ padding: '6px 14px', cursor: disabled ? 'not-allowed' : 'pointer' }}>
        {children}
      </button>
      {disabled && hovered && tooltip && (
        <span style={{
          position: 'absolute', bottom: 'calc(100% + 6px)', left: '50%', transform: 'translateX(-50%)',
          background: 'var(--surface-3)', border: '1px solid var(--border)', borderRadius: 'var(--radius)',
          padding: '4px 8px', fontSize: '11px', color: 'var(--text)', whiteSpace: 'nowrap',
          pointerEvents: 'none', zIndex: 10,
        }}>{tooltip}</span>
      )}
    </span>
  )
}

const pickFirstModelEntry = (payload: Record<string, unknown>) => {
  // payload = { computed_at, [model]: { [gen_mode]: { hard?, redirect? } } }
  for (const [key, value] of Object.entries(payload)) {
    if (key !== 'computed_at' && value && typeof value === 'object') {
      const genModes = value as Record<string, Record<string, ModeDivergence>>
      const firstGenMode = Object.values(genModes)[0]
      if (firstGenMode) return { modelId: key, modeMap: firstGenMode }
    }
  }
  return null
}

export const AblationPanel = ({ runId, models }: AblationPanelProps) => {
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
  const [modelId, setModelId]     = useState<string>()
  const [modelLoad, setModelLoad] = useState<ModelLoadState>('idle')

  useEffect(() => {
    getDivergence(runId)
      .then(async payload => {
        const entry = pickFirstModelEntry(payload as Record<string, unknown>)
        if (!entry) { setError('No divergence data for this run'); setStatus('idle'); return }
        const { modelId: foundModelId, modeMap } = entry
        setModelId(foundModelId)
        setHard(modeMap.hard)
        setRedirect(modeMap.redirect)
        const reference = modeMap.hard ?? modeMap.redirect
        if (reference) {
          setOnset(reference.suggested_onset)
          setSplit(reference.suggested_divergence)
        }
        setStatus('idle')

        const modelInfo = models.find(model => model.modelId === foundModelId)
        if (!modelInfo) { setModelLoad('error'); setError(`Model ${foundModelId} not in registry`); return }

        try {
          const { loaded_model } = await inferenceStatus()
          if (loaded_model === foundModelId) { setModelLoad('ready'); return }
          setModelLoad('loading')
          await inferenceLoad({ model_id: foundModelId, api_model_id: modelInfo.apiModelId })
          setModelLoad('ready')
        } catch (err) {
          setModelLoad('error')
          setError(`Model load failed: ${ String((err as Error).message) }`)
        }
      })
      .catch(err => { setError(String(err.message)); setStatus('idle') })
  }, [runId, models])

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
          {modelLoad === 'loading' && (
            <div style={{ color: 'var(--text-muted)', fontSize: '12px' }}>
              <span className="spinner" /> Model loading: <code>{modelId}</code>
            </div>
          )}
          {modelLoad === 'ready' && (
            <div style={{ color: 'var(--accent)', fontSize: '12px' }}>
              Model loaded: <code>{modelId}</code>
            </div>
          )}
          <div style={{ display: 'flex', gap: '12px' }}>
            <GatedButton disabled={modelLoad !== 'ready'}
              tooltip={modelLoad === 'loading' ? 'Model loading…' : 'Model not loaded'}
              onClick={() => applyAblation(runId).then(() => setStatus('applied')).catch(err => setError(String(err.message)))}>
              Apply hooks
            </GatedButton>
            <GatedButton disabled={modelLoad !== 'ready'}
              tooltip={modelLoad === 'loading' ? 'Model loading…' : 'Model not loaded'}
              onClick={() => clearAblation(runId).then(() => setStatus('idle')).catch(err => setError(String(err.message)))}>
              Clear hooks
            </GatedButton>
            <GatedButton disabled={verifying || modelLoad !== 'ready'}
              tooltip={modelLoad === 'loading' ? 'Model loading…' : 'Model not loaded'}
              onClick={runVerify}>
              {verifying ? 'Verifying…' : 'Verify'}
            </GatedButton>
            <GatedButton disabled={baking || modelLoad !== 'ready'}
              tooltip={modelLoad === 'loading' ? 'Model loading…' : 'Model not loaded'}
              onClick={runBake}>
              {baking ? 'Baking…' : 'Bake & save model'}
            </GatedButton>
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
