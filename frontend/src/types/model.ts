export interface LLM {
  _id: string
  name: string
  modelId: string
  apiModelId: string
  vendor: string
  architecture: {
    numLayers: number
    hiddenSize: number
  }
}
