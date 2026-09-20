import { PencilIcon } from '../atoms/PencilIcon'

// Pencil button — dim, rounded-square border on hover.
export const EditButton = ({ onClick }: { onClick: () => void }) => (
  <button
    onClick={onClick}
    title="Reclassify"
    className="audit-edit"
    style={{
      display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
      width: '24px', height: '24px', padding: 0, margin: 0,
      background: 'transparent', color: 'var(--text-muted)', border: '1px solid transparent',
      borderRadius: '6px', cursor: 'pointer', transition: 'color .12s, border-color .12s',
    }}
  >
    <PencilIcon />
  </button>
)
