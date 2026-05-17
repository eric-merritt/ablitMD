import { useState } from 'react'
import { inferenceLoad, inferenceGenerate, inferenceCompute } from '../api/inference'
import type { Run, ModeDirectionResult } from '../types/run'

export const useInference = () => {
  const [loadedModel, setLoadedModel] = useState<string | null>(null)
  const [modelLoading, setModelLoading] = useState(false)
  const [generating, setGenerating] = useState(false)
  const [computing, setComputing] = useState(false)

  const ensureModelLoaded = async (modelId: string, apiModelId: string) => {
    if (loadedModel === modelId) return
    setModelLoading(true)
    await inferenceLoad({ model_id: modelId, api_model_id: apiModelId })
    setLoadedModel(modelId)
    setModelLoading(false)
  }

  const generate = async (body: {
    prompt_id: string
    prompt_text: string
    run_id: string
    model_id: string
    mode: string
  }) => {
    setGenerating(true)
    try {
      return await inferenceGenerate(body)
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
