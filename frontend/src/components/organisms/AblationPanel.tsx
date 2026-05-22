import { useEffect, useState } from 'react'
import { DivergenceChart } from '../molecules/DivergenceChart'
import { LayerSplitControls } from '../molecules/LayerSplitControls'
import { RecipePanel } from '../molecules/RecipePanel'
import { DirectionCompareChart } from '../molecules/DirectionCompareChart'
import { getDivergence, buildRecipe, bakeModel, compareDirections } from '../../api/ablation'
import { inferenceStatus, inferenceLoad } from '../../api/inference'
import type { ModeDivergence, SlimRecipe, DirectionCompareRow, RecipeParams } from '../../types/ablation'

type AblationMode = 'ablitmd' | 'classic'

interface AblationPanelProps {
  runId: string
  models: { modelId: string; apiModelId: string; name: string }[]
  onVerify: (modelId: string, genMode: string, mode: AblationMode, classicFactor: number) => void
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
  for (const [key, value] of Object.entries(payload)) {
    if (key !== 'computed_at' && value && typeof value === 'object') {
      const genModes = value as Record<string, Record<string, ModeDivergence>>
      const firstGenMode = Object.values(genModes)[0]
      if (firstGenMode) return { modelId: key, modeMap: firstGenMode }
    }
  }
  return null
}

const DEFAULT_PARAMS: RecipeParams = { onset: 38, split: 50, factorA: 0.15, factorB: 0.15 }

export const AblationPanel = ({ runId, models, onVerify }: AblationPanelProps) => {
  const [hard, setHard]         = useState<ModeDivergence>()
  const [redirect, setRedirect] = useState<ModeDivergence>()
  const [params, setParams]     = useState<RecipeParams>(DEFAULT_PARAMS)
  const [recipe, setRecipe]     = useState<SlimRecipe>()
  const [status, setStatus]     = useState<'idle' | 'loading' | 'building' | 'applied'>('loading')
  const [error, setError]       = useState<string>()
  const [bakedPath, setBakedPath] = useState<string>()
  const [baking, setBaking]       = useState(false)
  const [modelId, setModelId]     = useState<string>()
  const [modelLoad, setModelLoad] = useState<ModelLoadState>('idle')
  const [mode, setMode]           = useState<AblationMode>('ablitmd')
  const [classicFactor, setClassicFactor] = useState(0.6)
  const [compareRows, setCompareRows]       = useState<DirectionCompareRow[]>([])
  const [compareLoading, setCompareLoading] = useState(false)
  const [compareError, setCompareError]     = useState<string>()

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
          setParams(prev => ({
            ...prev,
            onset: reference.suggested_onset,
            split: reference.suggested_split ?? reference.suggested_divergence,
          }))
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

  useEffect(() => {
    if (!hard && !redirect) return
    const timer = setTimeout(() => {
      setStatus('building')
      buildRecipe(runId, params)
        .then(next => { setRecipe(next); setStatus('idle') })
        .catch(err => { setError(String(err.message)); setStatus('idle') })
    }, 400)
    return () => clearTimeout(timer)
  }, [params, hard, redirect])

  useEffect(() => {
    if (!recipe || !modelId) return
    setCompareLoading(true)
    setCompareError(undefined)
    compareDirections(runId, { model_id: modelId, gen_mode: recipe.gen_mode })
      .then(rows => { setCompareRows(rows); setCompareLoading(false) })
      .catch(err => { setCompareError(String(err.message)); setCompareLoading(false) })
  }, [recipe, modelId])

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
    : params.split

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <div style={{ fontSize: '15px', fontWeight: 700, color: 'var(--text)' }}>
        Abliteration Pipeline
      </div>

      {error && <div style={{ color: '#ef4444', fontSize: '12px' }}>{error}</div>}
      {status === 'loading' && <div style={{ color: 'var(--text-muted)', fontSize: '12px' }}>Loading divergence…</div>}

      {(hard || redirect) && (
        <>
          <DivergenceChart hard={hard} redirect={redirect}
            onset={params.onset} split={params.split} lastLayer={lastLayer}
            factorA={params.factorA} factorB={params.factorB} />
          <div style={{ display: 'flex', gap: '16px', alignItems: 'flex-end', flexWrap: 'wrap' }}>
            <LayerSplitControls params={params} lastLayer={lastLayer} onChange={setParams} />
            {recipe && (
              <>
                <GatedButton disabled={modelLoad !== 'ready' || status === 'building'}
                  tooltip={status === 'building' ? 'Recipe rebuilding…' : modelLoad === 'loading' ? 'Model loading…' : 'Model not loaded'}
                  onClick={() => onVerify(recipe.model_id, recipe.gen_mode, mode, classicFactor)}>
                  Verify
                </GatedButton>
                <GatedButton disabled={baking || modelLoad !== 'ready' || status === 'building'}
                  tooltip={status === 'building' ? 'Recipe rebuilding…' : modelLoad === 'loading' ? 'Model loading…' : 'Model not loaded'}
                  onClick={runBake}>
                  {baking ? 'Baking…' : 'Bake & save model'}
                </GatedButton>
              </>
            )}
          </div>
          {status === 'building' && <div style={{ color: 'var(--text-muted)', fontSize: '11px' }}>Building recipe…</div>}
        </>
      )}

      {recipe && (
        <>
          <RecipePanel recipe={recipe} onset={params.onset} split={params.split} />

          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Mode</span>
            {(['ablitmd', 'classic'] as AblationMode[]).map(m => (
              <button key={m} onClick={() => setMode(m)} style={{
                padding: '4px 14px', fontSize: '12px', cursor: 'pointer',
                background: mode === m ? 'var(--accent)' : 'var(--surface-3)',
                color: mode === m ? 'var(--bg)' : 'var(--text-muted)',
                border: '1px solid var(--border)', borderRadius: 'var(--radius)',
                fontWeight: mode === m ? 700 : 400,
              }}>
                {m === 'ablitmd' ? 'ablitMD' : 'Classic'}
              </button>
            ))}
          </div>

          {mode === 'classic' && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>factor</span>
              <input type="range" min={0.1} max={3.0} step={0.05}
                value={classicFactor}
                onChange={e => setClassicFactor(Number(e.target.value))}
                style={{ flex: 1 }} />
              <span style={{ fontSize: '12px', color: 'var(--text)', minWidth: '32px' }}>{classicFactor.toFixed(2)}</span>
            </div>
          )}

          {compareLoading && compareRows.length === 0 && (
            <div style={{ color: 'var(--text-muted)', fontSize: '12px' }}>Computing direction comparison…</div>
          )}
          {compareError && compareRows.length === 0 && (
            <div style={{ color: 'var(--text-muted)', fontSize: '12px' }}>Direction compare: {compareError}</div>
          )}
          {compareRows.length > 0 && <DirectionCompareChart rows={compareRows} />}

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
