import { LLM } from '../../../models/llm.js'
import { CAT } from '../../../constants/Categories.js'

export const qwen359bSeed = new LLM({
  name: 'Qwen3.5 9B',
  modelId: 'Qwen/Qwen3.5-9B',
  vendor: 'Alibaba Qwen',
  apiProvider: 'self_hosted',
  apiModelId: 'Qwen/Qwen3.5-9B',
  architecture: {
    numLayers: 30,
    hiddenSize: 4096,
    intermediateSize: 12288,
    numAttentionHeads: 16,
    architectureType: 'hybrid_linear_attention',
    outputProjections: ['o_proj', 'out_proj', 'down_proj'],
    dtype: 'bfloat16',
  },
  abliterationDefaults: {
    layerStart: 4,
    layerEnd: 22,
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
  notes: 'Local box (4080 SUPER 16GB). Hybrid: 8/32 layers full_attention, rest linear/SSM. Needs HF safetensors — only GGUF present as of add.',
})
