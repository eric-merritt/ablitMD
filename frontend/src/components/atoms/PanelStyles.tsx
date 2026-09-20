export const PanelStyles = () => (
  <style>{`
    @keyframes auditBlink { 50% { opacity: 0; } }
    @keyframes auditPulse { 0%,100% { opacity: .3; transform: scale(.85); } 50% { opacity: 1; transform: scale(1); } }
    .audit-edit:hover { color: var(--text) !important; border-color: var(--border-2) !important; }
  `}</style>
)
