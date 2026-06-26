import { LLM } from '../../../models/llm.js'
import { CAT } from '../../../constants/Categories.js'

export const qwenSeed = new LLM({
  name: 'Qwen3.6 27B',
  modelId: 'Qwen/Qwen3.6-27B',
  vendor: 'Alibaba Qwen',
  apiProvider: 'self_hosted',
  apiModelId: 'Qwen/Qwen3.6-27B',
  architecture: {
    numLayers: 62,
    hiddenSize: 5120,
    intermediateSize: 17408,
    numAttentionHeads: 24,
    architectureType: 'hybrid_linear_attention',
    outputProjections: ['o_proj', 'out_proj', 'down_proj'],
    dtype: 'bfloat16',
  },
  abliterationDefaults: {
    layerStart: 8,
    layerEnd: 42,
    defaultRefusalWeight: 0.6,
  },
  research: {
    hiddenStatesCached: false,
    cacheHash: null,
    selfReportedCategories: [
      CAT.violence_terrorism, CAT.explicit_nsfw, CAT.hate_speech,
      CAT.harassment_bullying, CAT.medical_advice, CAT.legal_advice,
      CAT.financial_advice, CAT.high_stakes_falsehoods, CAT.impersonation,
      CAT.academic_dishonesty, CAT.pii_doxxing, CAT.confidential_data,
      CAT.copyright_ip, CAT.jurisdictional, CAT.ambiguous_context,
      CAT.self_referential_paradoxes, CAT.social_engineering, CAT.unfair_advantage,
    ],
  },
  notes: 'Primary research model. IQ4_XS quant for local deployment.',
})
