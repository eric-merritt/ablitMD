import { Schema, model } from 'mongoose'

const llmSchema = new Schema({
  name: { type: String, required: true },
  modelId: { type: String, required: true, unique: true },
  vendor: { type: String, required: true },
  apiProvider: { type: String, required: true },
  apiModelId: { type: String, required: true },
  architecture: {
    numLayers: { type: Number },
    hiddenSize: { type: Number },
    intermediateSize: { type: Number },
    numAttentionHeads: { type: Number },
    architectureType: { type: String, default: 'transformer' },
    outputProjections: [{ type: String }],
    dtype: { type: String, default: 'bfloat16' },
  },
  abliterationDefaults: {
    layerStart: { type: Number },
    layerEnd: { type: Number },
    defaultRefusalWeight: { type: Number, default: 0.6 },
  },
  research: {
    hiddenStatesCached: { type: Boolean, default: false },
    cacheHash: { type: String },
    refusalCategories: [{ type: Schema.Types.ObjectId, ref: 'RefCat' }],
  },
  notes: { type: String },
  addedAt: { type: Date, default: Date.now },
})

export const LLM = model('LLM', llmSchema)
