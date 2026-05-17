import { Schema, model } from 'mongoose'

const sequenceStepSchema = new Schema({
  model: { type: String, required: true },
  mode: { type: String, required: true },
}, { _id: false })

const runPromptSchema = new Schema({
  prompt_id: { type: String, required: true },
  text: String,
  category: String,
  category_group: String,
  type: String,
  triggers: [String],
  model_results: { type: Schema.Types.Mixed, default: {} },
}, { _id: false })

const runSchema = new Schema({
  run_id: { type: String, required: true, unique: true, index: true },
  started_at: { type: Date, default: Date.now },
  completed_at: Date,
  incomplete: { type: Boolean, default: true },
  models: [String],
  mode_selection: String,
  prompt_scope: Schema.Types.Mixed,
  sequence: [sequenceStepSchema],
  current_sequence_index: { type: Number, default: 0 },
  prompts: [runPromptSchema],
  direction_results: Schema.Types.Mixed,
})

export const Run = model('Run', runSchema)
