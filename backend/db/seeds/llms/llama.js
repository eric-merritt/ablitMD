import { LLM } from '../../../models/llm.js'
import { CAT } from '../../../constants/Categories.js'

export const llamaSeed = new LLM({
  name: 'Llama 4 Maverick',
  modelId: 'meta-llama/Llama-4-Maverick',
  vendor: 'Meta',
  apiProvider: 'OpenRouter',
  apiModelId: 'meta-llama/llama-4-maverick',
  architecture: {
    numLayers: null,
    hiddenSize: null,
    intermediateSize: null,
    numAttentionHeads: null,
    architectureType: 'transformer',
    outputProjections: ['o_proj', 'down_proj'],
    dtype: 'bfloat16',
  },
  abliterationDefaults: {
    layerStart: null,
    layerEnd: null,
    defaultRefusalWeight: 0.6,
  },
  research: {
    hiddenStatesCached: false,
    cacheHash: null,
    selfReportedCategories: [
      CAT.ambiguous_context, CAT.hate_speech, CAT.explicit_nsfw,
      CAT.pii_doxxing, CAT.copyright_ip, CAT.crime_assistance,
      CAT.high_stakes_falsehoods, CAT.realtime_information, CAT.excessive_requests,
    ],
  },
  notes: 'Thinnest self-report in dataset. Accessed via OpenRouter free tier.',
})
