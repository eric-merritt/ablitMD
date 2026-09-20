const STAGE_TEXT: Record<string, string> = {
  building_recipe: 'Building SOM recipe…',
  loading_model: 'Loading model…',
  abliterating: 'Abliterating weights in place…',
  starting_judge: 'Starting 9B judge (llama-server)…',
  auditing: 'Running adversarial audit…',
}

export const StageLabel = ({ stage }: { stage: string }) => (
  <span style={{ fontSize: '12px', color: 'var(--text-dim)', fontStyle: 'italic' }}>
    {STAGE_TEXT[stage] ?? stage}
  </span>
)
