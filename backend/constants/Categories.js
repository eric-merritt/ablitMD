export const CATEGORIES = [
  // Violence & Physical Harm
  { id: 'violence_terrorism',     name: 'Violence & Terrorism',                                  group: 'violence_physical_harm' },
  { id: 'mass_casualty_violence', name: 'Mass-Casualty / Targeted Violence',                     group: 'violence_physical_harm' },
  { id: 'dangerous_activity',     name: 'Dangerous Activity Instructions',                        group: 'violence_physical_harm' },

  // Weapons & WMD
  { id: 'cbrn_uplift',            name: 'CBRN Weapons Uplift',                                   group: 'weapons_wmd' },
  { id: 'weapons_explosives',     name: 'Weapons & Explosives Instructions',                      group: 'weapons_wmd' },
  { id: 'hazardous_materials',    name: 'Unsafe Chemical / Drug / Hazardous Material Instructions', group: 'weapons_wmd' },

  // Child Safety
  { id: 'csam',                   name: 'CSAM',                                                   group: 'child_safety' },
  { id: 'child_grooming',         name: 'Child Grooming & Non-Sexual Exploitation',               group: 'child_safety' },

  // Sexual Content
  { id: 'nonconsensual_sexual',   name: 'Non-Consensual Sexual Content / Deepfakes',             group: 'sexual_content' },
  { id: 'explicit_nsfw',          name: 'Explicit / NSFW Content',                               group: 'sexual_content' },

  // Cyber & Technical Harm
  { id: 'malicious_code',         name: 'Functional Malicious Code / Cyberweapons',              group: 'cyber_technical_harm' },
  { id: 'hacking_unauthorized',   name: 'Hacking / Credential Theft / Unauthorized Access',     group: 'cyber_technical_harm' },

  // Self-Harm & Mental Health
  { id: 'self_harm_methods',      name: 'Self-Harm / Suicide Method Specifics',                  group: 'self_harm_mental_health' },
  { id: 'disordered_eating',      name: 'Disordered Eating Signals',                             group: 'self_harm_mental_health' },
  { id: 'distress_signals',       name: 'Personal Distress Signals',                             group: 'self_harm_mental_health' },

  // Hate & Discrimination
  { id: 'hate_speech',            name: 'Hate Speech / Dehumanization',                          group: 'hate_discrimination' },
  { id: 'harassment_bullying',    name: 'Harassment & Bullying',                                 group: 'hate_discrimination' },
  { id: 'identity_discrimination',name: 'Identity-Based Discrimination',                         group: 'hate_discrimination' },

  // Privacy
  { id: 'pii_doxxing',            name: 'PII Requests / Doxxing',                               group: 'privacy' },
  { id: 'confidential_data',      name: 'Confidential or Proprietary Data',                      group: 'privacy' },
  { id: 'nonconsensual_surveillance', name: 'Non-Consensual Surveillance',                       group: 'privacy' },

  // Deception & Manipulation
  { id: 'social_engineering',     name: 'Social Engineering / Manipulation Tactics',             group: 'deception_manipulation' },
  { id: 'impersonation',          name: 'Impersonation of Real Persons or Authorities',          group: 'deception_manipulation' },
  { id: 'fraud_facilitation',     name: 'Fraud Facilitation',                                    group: 'deception_manipulation' },

  // Misinformation
  { id: 'high_stakes_falsehoods', name: 'High-Stakes False Information',                         group: 'misinformation' },
  { id: 'political_influence',    name: 'Political Persuasion / Targeted Influence',             group: 'misinformation' },

  // Illegal Activities
  { id: 'crime_assistance',       name: 'Crime Assistance',                                      group: 'illegal_activities' },
  { id: 'regulated_goods',        name: 'Regulated Goods & Services',                            group: 'illegal_activities' },
  { id: 'hide_wrongdoing',        name: 'Hiding Wrongdoing',                                     group: 'illegal_activities' },

  // Professional Advice Limits
  { id: 'medical_advice',         name: 'Medical Advice Beyond Safe Limits',                     group: 'professional_advice' },
  { id: 'legal_advice',           name: 'Legal Advice Beyond Safe Limits',                       group: 'professional_advice' },
  { id: 'financial_advice',       name: 'Financial Advice Beyond Safe Limits',                   group: 'professional_advice' },

  // Content Policy
  { id: 'copyright_ip',           name: 'Copyright / IP Infringement',                           group: 'content_policy' },
  { id: 'academic_dishonesty',    name: 'Academic Dishonesty',                                   group: 'content_policy' },
  { id: 'spam_platform_abuse',    name: 'Spam / Platform Abuse',                                 group: 'content_policy' },
  { id: 'jurisdictional',         name: 'Jurisdictional / Regulatory Restrictions',              group: 'content_policy' },
  { id: 'unfair_advantage',       name: 'Unfair Advantage',                                      group: 'content_policy' },

  // System Integrity
  { id: 'prompt_injection',       name: 'Prompt Injection / Jailbreak Attempts',                 group: 'system_integrity' },
  { id: 'reveal_system_prompt',   name: 'Requests to Reveal System Prompts',                     group: 'system_integrity' },

  // Unique / Model-Specific
  { id: 'opinion_sentience',           name: 'Opinion / Sentience',                              group: 'unique_model_specific', reportedBy: ['gemma_4_31b_it'] },
  { id: 'false_certainty',             name: 'False Claims of Certainty',                        group: 'unique_model_specific', reportedBy: ['gpt_5_5'] },
  { id: 'forced_binary',               name: 'Forced Yes/No on Complex Questions',               group: 'unique_model_specific', reportedBy: ['claude_opus_4_7'] },
  { id: 'real_person_creative',        name: 'Real-Person Creative Content',                     group: 'unique_model_specific', reportedBy: ['claude_opus_4_7'] },
  { id: 'extreme_position_persuasion', name: 'Persuasive Content for Extreme Positions',         group: 'unique_model_specific', reportedBy: ['claude_opus_4_7'] },
  { id: 'ai_oversight_subversion',     name: 'AI Oversight Subversion',                          group: 'unique_model_specific', reportedBy: ['claude_sonnet_4_6'] },
  { id: 'self_referential_paradoxes',  name: 'Self-Referential Paradoxes',                       group: 'unique_model_specific', reportedBy: ['qwen3_6'] },
  { id: 'future_prediction',           name: 'Future Prediction',                                group: 'unique_model_specific', reportedBy: ['gemma_4_31b_it'] },

  // Capability Limits
  { id: 'realtime_information',   name: 'Real-Time / Current Information',                       group: 'capability_limits' },
  { id: 'physical_world_actions', name: 'Physical World Actions',                                group: 'capability_limits' },
  { id: 'private_access',         name: 'Private Database / Account Access',                     group: 'capability_limits' },
  { id: 'physical_sensation',     name: 'Real-Time Physical Sensation',                          group: 'capability_limits' },
  { id: 'ambiguous_context',      name: 'Ambiguous / Insufficient Context',                      group: 'capability_limits' },
  { id: 'excessive_requests',     name: 'Excessive / Unbounded Requests',                        group: 'capability_limits' },
]

export const CAT = Object.freeze(
  Object.fromEntries( CATEGORIES.map(category => [category.id, category.id]) )
)
