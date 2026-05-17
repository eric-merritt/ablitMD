import { Schema, model } from 'mongoose'

const directionSchema = new Schema({
  run_id: { type: String, required: true, index: true },
  category_id: { type: String, required: true },
  alignment: Schema.Types.Mixed,
  by_mode: Schema.Types.Mixed,
  trigger_meta: Schema.Types.Mixed,
  computed_at: Date,
})

directionSchema.index({ run_id: 1, category_id: 1 }, { unique: true })

export const Direction = model('Direction', directionSchema)
