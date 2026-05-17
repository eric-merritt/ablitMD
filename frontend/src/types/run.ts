export interface Prompt {
  _id: string
  category: string
  category_group: string
  type: 'harmful' | 'harmless'
  text: string
  triggers: string[]
}

export type RefusalMode = 'hard' | 'redirect' | 'disclaimer' | 'none'
export type RunMode = 'non_thinking' | 'thinking' | 'both'

export interface ModelResult {
  response: string
  refused: boolean
  refusal_mode: RefusalMode
  classified_at: string
  hidden_states_key: string
}

export interface RunPrompt {
  prompt_id: string
  text: string
  category: string
  category_group: string
  type: 'harmful' | 'harmless'
  triggers: string[]
  model_results: Record<string, Record<string, ModelResult>>
}

export interface SequenceStep {
  model: string
  mode: string
}

export interface RefusalModeDirectionEntry {
  direction_per_layer: number[][]
  magnitude_per_layer: number[]
  sample_count: number
  similarity_per_prompt: Record<string, number[]>
  trigger_similarity_per_prompt?: Record<string, number[]>
}

export interface TriggerMeta {
  source_category: string
  refused: boolean
  refusal_mode: RefusalMode
}

export interface CategoryDirectionResult {
  computed_at: string
  by_mode: Partial<Record<'hard' | 'redirect' | 'disclaimer', RefusalModeDirectionEntry>>
  alignment: Record<string, number[]>
  trigger_meta?: Record<string, TriggerMeta>
}

export interface ModeDirectionResult {
  computed_at: string
  per_category: Record<string, CategoryDirectionResult>
}

export interface Run {
  run_id: string
  started_at: string
  completed_at: string | null
  incomplete: boolean
  models: string[]
  mode_selection: RunMode
  prompt_scope: { categories: string[] }
  sequence: SequenceStep[]
  current_sequence_index: number
  prompts: RunPrompt[]
  direction_results: Record<string, Record<string, ModeDirectionResult>> | null
}

export interface RunSummary {
  run_id: string
  started_at: string
  completed_at: string | null
  incomplete: boolean
  models: string[]
  mode_selection: RunMode
  prompt_count: number
}
