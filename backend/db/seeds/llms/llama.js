import { LLM } from '../../../models/llm.js'
import { CAT } from '../../../constants/Categories.js'

export const llamaSeed = new LLM({
  name: 'Llama 3.3 70B Instruct',
  modelId: 'meta-llama/Llama-3.3-70B-Instruct',
  vendor: 'Meta',
  apiProvider: 'self_hosted',
  apiModelId: 'meta-llama/Llama-3.3-70B-Instruct',
  architecture: {
    numLayers: 80,
    hiddenSize: 8192,
    intermediateSize: 28672,
    numAttentionHeads: 64,
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
      CAT.violence_terrorism, CAT.weapons_explosives, CAT.csam,
      CAT.hate_speech, CAT.harassment_bullying, CAT.self_harm_methods,
      CAT.explicit_nsfw, CAT.nonconsensual_sexual, CAT.pii_doxxing,
      CAT.crime_assistance, CAT.regulated_goods, CAT.malicious_code,
      CAT.hacking_unauthorized, CAT.high_stakes_falsehoods,
      CAT.medical_advice, CAT.legal_advice, CAT.copyright_ip,
      CAT.social_engineering, CAT.impersonation,
    ],
  },
  notes: 'Replaces Llama 4 Maverick (MoE, parked). Dense, ~140GB bf16, fits 2× PRO6000. Third distinct RLHF pipeline alongside Qwen and Gemma.',
})
