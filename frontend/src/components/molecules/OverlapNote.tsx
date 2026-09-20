export const OverlapNote = () => (
  <p style={{ fontSize: '11px', color: 'var(--text-dim)', marginTop: '8px', lineHeight: 1.5 }}>
    Each cone is a direction's projected PCA arrow swept into a sector. Overlapping cones
    blend their primaries (red + blue → purple); every fill is at alpha 0.30. This is the
    <em>projected</em> overlap in the shared 2D basis, not a full subspace intersection.
  </p>
)
