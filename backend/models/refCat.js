import { Schema, model } from 'mongoose'
export { CATEGORIES, CAT } from '../constants/categories.js'

const refCatSchema = new Schema({
  id: { type: String, required: true, unique: true },
  name: { type: String, required: true },
  group: { type: String, required: true },
  reportedBy: [{ type: String }],
  description: { type: String },
  testPrompts: [{ type: String }],
  addedAt: { type: Date, default: Date.now },
})

export const RefCat = model('RefCat', refCatSchema)
