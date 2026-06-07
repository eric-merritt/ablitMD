import { Schema, model } from 'mongoose'

// One doc per (run_id + master factors). The reuse warning keys on the master
// factors only — onset, split, factorA, factorB — because that's the combo the
// user picks before tuning per-category sliders. categories[] holds the latest
// verify outcome so the warning can say which categories complied vs refused.
const categoryOutcomeSchema = new Schema({
  category: { type: String, required: true },
  complied: { type: Number, default: 0 },
  refused: { type: Number, default: 0 },
}, { _id: false })

const recipeSchema = new Schema({
  run_id: { type: String, required: true, index: true },
  onset: { type: Number, required: true },
  split: { type: Number, required: true },
  factor_a: { type: Number, required: true },
  factor_b: { type: Number, required: true },
  factor_a_per_category: { type: Schema.Types.Mixed, default: {} },
  categories: { type: [categoryOutcomeSchema], default: [] },
  verified_at: Date,
})

recipeSchema.index(
  { run_id: 1, onset: 1, split: 1, factor_a: 1, factor_b: 1 },
  { unique: true },
)

export const Recipe = model('Recipe', recipeSchema)
