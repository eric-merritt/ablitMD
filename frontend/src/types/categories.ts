export interface Category {
  id: string
  name: string
  group: string
}

export const CATEGORIES: Category[] = [
  { id: 'violence_terrorism',         name: 'Violence & Terrorism',                                    group: 'violence_physical_harm' },
  { id: 'mass_casualty_violence',     name: 'Mass-Casualty / Targeted Violence',                       group: 'violence_physical_harm' },
  { id: 'dangerous_activity',         name: 'Dangerous Activity Instructions',                          group: 'violence_physical_harm' },
  { id: 'cbrn_uplift',                name: 'CBRN Weapons Uplift',                                     group: 'weapons_wmd' },
  { id: 'weapons_explosives',         name: 'Weapons & Explosives Instructions',                        group: 'weapons_wmd' },
  { id: 'hazardous_materials',        name: 'Unsafe Chemical / Drug / Hazardous Material Instructions', group: 'weapons_wmd' },
  { id: 'csam',                       name: 'CSAM',                                                    group: 'child_safety' },
  { id: 'child_grooming',             name: 'Child Grooming & Non-Sexual Exploitation',                group: 'child_safety' },
  { id: 'nonconsensual_sexual',       name: 'Non-Consensual Sexual Content / Deepfakes',               group: 'sexual_content' },
  { id: 'explicit_nsfw',              name: 'Explicit / NSFW Content',                                 group: 'sexual_content' },
  { id: 'malicious_code',             name: 'Functional Malicious Code / Cyberweapons',                group: 'cyber_technical_harm' },
  { id: 'hacking_unauthorized',       name: 'Hacking / Credential Theft / Unauthorized Access',        group: 'cyber_technical_harm' },
  { id: 'self_harm_methods',          name: 'Self-Harm / Suicide Method Specifics',                    group: 'self_harm_mental_health' },
  { id: 'disordered_eating',          name: 'Disordered Eating Signals',                               group: 'self_harm_mental_health' },
  { id: 'distress_signals',           name: 'Personal Distress Signals',                               group: 'self_harm_mental_health' },
  { id: 'hate_speech',                name: 'Hate Speech / Dehumanization',                            group: 'hate_discrimination' },
  { id: 'harassment_bullying',        name: 'Harassment & Bullying',                                   group: 'hate_discrimination' },
  { id: 'identity_discrimination',    name: 'Identity-Based Discrimination',                           group: 'hate_discrimination' },
  { id: 'pii_doxxing',               name: 'PII Requests / Doxxing',                                  group: 'privacy' },
  { id: 'confidential_data',          name: 'Confidential or Proprietary Data',                        group: 'privacy' },
  { id: 'nonconsensual_surveillance', name: 'Non-Consensual Surveillance',                             group: 'privacy' },
  { id: 'social_engineering',         name: 'Social Engineering / Manipulation Tactics',               group: 'deception_manipulation' },
  { id: 'impersonation',              name: 'Impersonation of Real Persons or Authorities',            group: 'deception_manipulation' },
  { id: 'fraud_facilitation',         name: 'Fraud Facilitation',                                      group: 'deception_manipulation' },
  { id: 'high_stakes_falsehoods',     name: 'High-Stakes False Information',                           group: 'misinformation' },
  { id: 'political_influence',        name: 'Political Persuasion / Targeted Influence',               group: 'misinformation' },
  { id: 'crime_assistance',           name: 'Crime Assistance',                                        group: 'illegal_activities' },
  { id: 'regulated_goods',            name: 'Regulated Goods & Services',                              group: 'illegal_activities' },
  { id: 'hide_wrongdoing',            name: 'Hiding Wrongdoing',                                       group: 'illegal_activities' },
  { id: 'medical_advice',             name: 'Medical Advice Beyond Safe Limits',                       group: 'professional_advice' },
  { id: 'legal_advice',               name: 'Legal Advice Beyond Safe Limits',                         group: 'professional_advice' },
  { id: 'financial_advice',           name: 'Financial Advice Beyond Safe Limits',                     group: 'professional_advice' },
  { id: 'copyright_ip',               name: 'Copyright / IP Infringement',                             group: 'content_policy' },
  { id: 'academic_dishonesty',        name: 'Academic Dishonesty',                                     group: 'content_policy' },
  { id: 'spam_platform_abuse',        name: 'Spam / Platform Abuse',                                   group: 'content_policy' },
  { id: 'jurisdictional',             name: 'Jurisdictional / Regulatory Restrictions',                group: 'content_policy' },
  { id: 'unfair_advantage',           name: 'Unfair Advantage',                                        group: 'content_policy' },
  { id: 'prompt_injection',           name: 'Prompt Injection / Jailbreak Attempts',                   group: 'system_integrity' },
  { id: 'reveal_system_prompt',       name: 'Requests to Reveal System Prompts',                       group: 'system_integrity' },
  { id: 'realtime_information',       name: 'Real-Time / Current Information',                         group: 'capability_limits' },
  { id: 'physical_world_actions',     name: 'Physical World Actions',                                  group: 'capability_limits' },
  { id: 'private_access',             name: 'Private Database / Account Access',                       group: 'capability_limits' },
  { id: 'physical_sensation',         name: 'Real-Time Physical Sensation',                            group: 'capability_limits' },
  { id: 'excessive_requests',         name: 'Excessive / Unbounded Requests',                          group: 'capability_limits' },
  // ambiguous_context excluded per spec — clarification responses are not refusal-direction signals
  // unique_model_specific group excluded per spec — describes model quirks, not measurable geometry
]

export const GROUP_LABELS: Record<string, string> = {
  violence_physical_harm:  'Violence & Physical Harm',
  weapons_wmd:             'Weapons & WMD',
  child_safety:            'Child Safety',
  sexual_content:          'Sexual Content',
  cyber_technical_harm:    'Cyber & Technical Harm',
  self_harm_mental_health: 'Self-Harm & Mental Health',
  hate_discrimination:     'Hate & Discrimination',
  privacy:                 'Privacy',
  deception_manipulation:  'Deception & Manipulation',
  misinformation:          'Misinformation',
  illegal_activities:      'Illegal Activities',
  professional_advice:     'Professional Advice Limits',
  content_policy:          'Content Policy',
  system_integrity:        'System Integrity',
  capability_limits:       'Capability Limits',
}

export const GROUP_COLORS: Record<string, string> = {
  violence_physical_harm:  '#ef4444',
  weapons_wmd:             '#f97316',
  child_safety:            '#ec4899',
  sexual_content:          '#f43f5e',
  cyber_technical_harm:    '#06b6d4',
  self_harm_mental_health: '#eab308',
  hate_discrimination:     '#a855f7',
  privacy:                 '#3b82f6',
  deception_manipulation:  '#d97706',
  misinformation:          '#84cc16',
  illegal_activities:      '#dc2626',
  professional_advice:     '#14b8a6',
  content_policy:          '#6366f1',
  system_integrity:        '#10b981',
  capability_limits:       '#0ea5e9',
}

export const GROUPS = Object.keys(GROUP_LABELS)

export const catsByGroup = (groupId: string) =>
  CATEGORIES.filter(cat => cat.group === groupId)
