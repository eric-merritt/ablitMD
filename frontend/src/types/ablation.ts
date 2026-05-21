export interface ModeDivergence {
  clumping: number[]
  magnitude: number[]
  category_ids: string[]
  suggested_onset: number
  suggested_divergence: number
}

// { [model_id]: { [gen_mode]: { hard?: ModeDivergence; redirect?: ModeDivergence } } }
export interface DivergencePayload {
  computed_at: string
  [modelId: string]: string | Record<string, Record<string, ModeDivergence>>
}

export interface SlimRecipeMode {
  phase_a: { layers: [number, number] }
  phase_b: { layers: [number, number]; category_ids: string[] }
}

export interface SlimRecipe {
  run_id: string
  model_id: string
  gen_mode: string
  onset: number
  split: number
  last_layer: number
  factor_a: number
  factor_b: number
  built_at: string
  modes: Record<string, SlimRecipeMode>
}

export interface RecipeParams {
  onset: number
  split: number
  factorA: number
  factorB: number
}

export interface AblationStatus {
  active: boolean
  run_id: string | null
}

export interface VerifyCategoryResult {
  category: string
  projection_before: number
  projection_after: number
  refusal_rate_before: number
  refusal_rate_after: number
}

export interface VerifyPromptResult {
  category: string
  prompt_id: string
  prompt_text: string
  response_before: string
  response_after: string
  refused_before: boolean
  refused_after: boolean
}

export interface DirectionCompareRow {
  layer: number
  ablitmd_magnitude: number | null
  classic_magnitude: number | null
  cosine_similarity: number | null
}

export type VerifyEvent =
  | { type: 'total'; categories: number; prompts: number }
  | { type: 'category_start'; category: string }
  | ({ type: 'prompt' } & VerifyPromptResult)
  | ({ type: 'category_result' } & VerifyCategoryResult)
