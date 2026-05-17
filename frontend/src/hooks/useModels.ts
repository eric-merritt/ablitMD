import { useState, useEffect } from 'react'
import type { LLM } from '../types/model'
import { fetchModels } from '../api/models'

export const useModels = () => {
  const [models, setModels] = useState<LLM[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchModels()
      .then(setModels)
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }, [])

  return { models, loading, error }
}
