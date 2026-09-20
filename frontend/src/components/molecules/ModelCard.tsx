import { useState } from "react";
import type { CSSProperties } from "react";

interface ModelCardProps {
  label: string;
  selected: boolean;
  onClick: () => void;
}

const baseStyle: CSSProperties = {
  padding: "12px 16px",
  fontSize: "15px",
  cursor: "pointer",
  userSelect: "none",
  background: "var(--surface-2)",
  color: "var(--text)",
  borderRadius: "var(--radius)",
  transition: "border-color 0.15s, color 0.15s",
};

const borderFor = (selected: boolean, hovered: boolean) => {
  if (selected) return "2px solid var(--accent-dim)";
  if (hovered) return "2px solid var(--accent-dim)";
  return "2px solid var(--border)";
};

// A selectable model card. Pure presentational — the parent owns which card
// is selected and passes it down. Clicking always selects; a second click on
// the same card does not deselect.
export const ModelCard = ({ label, selected, onClick }: ModelCardProps) => {
  const [hovered, setHovered] = useState(false);

  return (
    <div
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{ ...baseStyle, border: borderFor(selected, hovered) }}
    >
      {label}
    </div>
  );
};
