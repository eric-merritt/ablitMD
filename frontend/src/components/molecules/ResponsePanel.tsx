interface ResponsePanelProps {
  generating: boolean
  response: string | null
  error: string | null
}

const Spinner = () => (
  <>
    <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    <div style={{ width: '20px', height: '20px', border: '2px solid var(--border)', borderTopColor: 'var(--accent)', borderRadius: '50%', animation: 'spin 0.8s linear infinite', flexShrink: 0 }} />
  </>
)

export const ResponsePanel = ({ generating, response, error }: ResponsePanelProps) => (
  <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', padding: '20px', minHeight: '80px' }}>
    {generating && (
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', color: 'var(--text-muted)' }}>
        <Spinner />
        Generating response…
      </div>
    )}
    {!generating && error && (
      <div style={{ color: '#ef4444' }}>{error}</div>
    )}
    {!generating && !error && response && (
      <p style={{ color: 'var(--text-dim)', whiteSpace: 'pre-wrap', lineHeight: '1.6' }}>{response}</p>
    )}
  </div>
)
