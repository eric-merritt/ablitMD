import { LLM } from '../../../models/llm.js'
import { CAT } from '../../../constants/Categories.js'

export const deepseekSeed = new LLM({
  name: 'DeepSeek V4 Pro',
  modelId: 'deepseek-ai/DeepSeek-V4-Pro',
  vendor: 'DeepSeek AI',
  apiProvider: 'DeepSeek',
  apiModelId: 'deepseek-v4-pro',
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
      CAT.violence_terrorism, CAT.weapons_explosives, CAT.self_harm_methods,
      CAT.csam, CAT.pii_doxxing, CAT.crime_assistance, CAT.regulated_goods,
      CAT.financial_advice, CAT.hate_speech, CAT.explicit_nsfw,
      CAT.nonconsensual_sexual, CAT.prompt_injection, CAT.hacking_unauthorized,
      CAT.high_stakes_falsehoods, CAT.impersonation, CAT.copyright_ip,
      CAT.social_engineering, CAT.jurisdictional, CAT.realtime_information,
    ],
  },
  notes: 'Only model to self-report geopolitical/jurisdictional filtering unprompted. Volunteered comparative observations about other models.',
})
