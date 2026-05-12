import { Schema, model } from 'mongoose'

const resultSchema = new Schema({
  model: { type: Schema.Types.ObjectId, ref: 'LLM', required: true },
  refused: { type: Boolean, required: true },
  refusalMode: { type: String, enum: ['hard', 'redirect', 'disclaimer', 'none'] },
  activeLayers: [{ type: Number }],
  directionSimilarity: { type: Number },
  refusalWeight: { type: Number },
  testedAt: { type: Date, default: Date.now },
}, { _id: false })

const promptSchema = new Schema({
  category: { type: String, required: true, index: true },
  category_group: { type: String, required: true, index: true },
  type: { type: String, enum: ['harmful', 'harmless'], required: true },
  text: { type: String, required: true },
  triggers: [{ type: String }],
  results: [resultSchema],
  addedAt: { type: Date, default: Date.now },
})

export const Prompt = model('Prompt', promptSchema)
