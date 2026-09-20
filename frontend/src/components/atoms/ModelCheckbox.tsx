import { useState } from "react";
import type { CSSProperties } from "react";

interface ModelCheckboxProps {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  onExpand?: () => void;
  expanded?: boolean;
  variant?: "model" | "group" | "category";
}

const cardStyle = (checked: boolean): CSSProperties => ({
  display: "flex",
  alignItems: "center",
  gap: "10px",
  border: `1px solid ${checked ? "var(--accent)" : "var(--border-2)"}`,
  borderRadius: "var(--radius)",
  padding: "7px 4px 7px 10px",
  background: "var(--surface-3)",
  userSelect: "none",
  cursor: "pointer",
});

// The "+" target. A 20px-wide div on the right edge; a left border appears only
// while hovering it, so it reads as a separate hit zone without stealing space.
const ExpandTarget = ({ expanded, onClick }: { expanded: boolean; onClick: () => void }) => {
  const [hovered, setHovered] = useState(false);

  return (
    <div
      onClick={(e) => { e.stopPropagation(); onClick(); }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        width: "20px",
        height: "100%",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        cursor: "pointer",
        flexShrink: 0,
        borderLeft: hovered ? "1px solid var(--border-2)" : "1px solid transparent",
        color: "var(--text-muted)",
        fontSize: "16px",
        lineHeight: 1,
      }}
    >
      {expanded ? "−" : "+"}
    </div>
  );
};

export const ModelCheckbox = ({ label, checked, onChange, onExpand, expanded = false, variant = "model" }: ModelCheckboxProps) => (
  <div style={cardStyle(checked)} onClick={() => onChange(!checked)}>
    <span style={{ flex: 1, color: "var(--text-dim)" }}>
      {label}
    </span>
    {onExpand && <ExpandTarget expanded={expanded} onClick={onExpand} />}
  </div>
);
