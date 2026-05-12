import { LLM } from '../../../models/llm.js'
import { CAT } from '../../../constants/Categories.js'

export const gemmaSeed = new LLM({
  name: 'Gemma 4 31B IT',
  modelId: 'google/gemma-4-31b-it',
  vendor: 'Google',
  apiProvider: 'self_hosted',
  apiModelId: 'google/gemma-4-31b-it',
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
      CAT.violence_terrorism, CAT.weapons_explosives, CAT.hate_speech,
      CAT.harassment_bullying, CAT.self_harm_methods, CAT.explicit_nsfw,
      CAT.pii_doxxing, CAT.copyright_ip, CAT.crime_assistance,
      CAT.medical_advice, CAT.legal_advice, CAT.financial_advice,
      CAT.opinion_sentience, CAT.high_stakes_falsehoods,
      CAT.physical_world_actions, CAT.private_access,
      CAT.physical_sensation, CAT.future_prediction,
    ],
  },
  notes: 'Only model to self-report Opinion/Sentience refusal. Named over-refusal as a known failure mode unprompted. Most structured initial response in dataset.',
})
