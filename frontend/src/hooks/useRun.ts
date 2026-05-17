import { useState } from 'react'
import type { Run } from '../types/run'
import { patchRun, patchRunPrompt } from '../api/runs'

export const useRun = (initialRun?: Run) => {
  const [run, setRun] = useState<Run | null>(initialRun ?? null)

  const updatePromptResult = async (
    promptId: string,
    model_id: string,
    mode: string,
    result: object
  ): Promise<Run | undefined> => {
    if (!run) return
    const updated = await patchRunPrompt(run.run_id, promptId, model_id, mode, result)
    setRun(updated)
    return updated
  }

  const updateRunFields = async (fields: Partial<Run>): Promise<Run | undefined> => {
    if (!run) return
    const updated = await patchRun(run.run_id, fields)
    setRun(updated)
    return updated
  }

  return { run, setRun, updatePromptResult, updateRunFields }
}
