import { useRef, useState } from 'react'
import { inferenceLoad, inferenceGenerateStream, inferenceCompute } from '../api/inference'
import type { Run, ModeDirectionResult } from '../types/run'

export const useInference = () => {
  const [loadedModel, setLoadedModel] = useState<string | null>(null)
  const [modelLoading, setModelLoading] = useState(false)
  const [generating, setGenerating] = useState(false)
  const [computing, setComputing] = useState(false)

  // Synchronous dedupe guard. The old loadedModel === modelId check read React
  // state, which is stale across two rapid calls (StrictMode double-invoke / re-render),
  // so both fired and we saw two POSTs to /load. A ref updates synchronously, so a
  // second call for the same in-flight or already-loaded model is a no-op.
  const loadGuard = useRef<{ id: string; promise: Promise<void> } | null>(null)

  const ensureModelLoaded = async (modelId: string, apiModelId: string) => {
    if (loadGuard.current && loadGuard.current.id === modelId) {
      return loadGuard.current.promise
    }
    setModelLoading(true)
    const promise = inferenceLoad({ model_id: modelId, api_model_id: apiModelId })
      .then(() => setLoadedModel(modelId))
      .finally(() => setModelLoading(false))
    loadGuard.current = { id: modelId, promise }
    return promise
  }

  const generate = async (
    body: {
      prompt_id: string
      prompt_text: string
      run_id: string
      model_id: string
      mode: string
    },
    onToken: (text: string) => void,
    signal?: AbortSignal,
  ): Promise<{ response: string; hidden_states_key: string }> => {
    setGenerating(true)
    let hidden_states_key = ''
    let finalResponse = ''
    let accumulated = ''
    try {
      await inferenceGenerateStream(body, (event) => {
        if (event.type === 'ready') hidden_states_key = event.hidden_states_key
        else if (event.type === 'token') { accumulated += event.text; onToken(event.text) }
        else if (event.type === 'done' || event.type === 'aborted') finalResponse = event.response
        else if (event.type === 'error') throw new Error(event.error)
      }, signal)
      return { response: finalResponse || accumulated, hidden_states_key }
    } finally {
      setGenerating(false)
    }
  }

  const computeAllDirections = async (
    run: Run
  ): Promise<Record<string, Record<string, ModeDirectionResult>>> => {
    setComputing(true)
    const direction_results: Record<string, Record<string, ModeDirectionResult>> = {}

    try {
      for (const step of run.sequence) {
        const per_category = await inferenceCompute({
          run_id: run.run_id,
          model_id: step.model,
          mode: step.mode,
        })
        direction_results[step.model] ??= {}
        direction_results[step.model][step.mode] = {
          computed_at: new Date().toISOString(),
          per_category,
        }
      }
    } finally {
      setComputing(false)
    }

    return direction_results
  }

  return { loadedModel, modelLoading, generating, computing, ensureModelLoaded, generate, computeAllDirections }
}
