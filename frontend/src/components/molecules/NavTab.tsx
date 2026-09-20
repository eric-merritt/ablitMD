import type { CSSProperties } from "react";

interface NavTabProps {
  active: boolean;
  label: string;
  onClick: () => void;
}

const Style = (active: boolean): CSSProperties => ({
  padding: "10px 20px",
  fontSize: "1.5rem",
  fontWeight: active ? 700 : 400,
  cursor: "pointer",
  background: "var(--surface)",
  color: active ? "var(--text)" : "var(--text-muted)",
  // Active tab has no bottom border and sits flush with the strip line so it
  // merges into the section below; inactive tabs keep their own bottom border.
  border: "1px solid var(--border)",
  borderBottom: active ? "none" : "1px solid var(--border)",
  borderRadius: "var(--radius) var(--radius) 0 0",
  marginBottom: active ? "-1px" : "0px",
  position: "relative",
  zIndex: active ? -1 : 0,
  transition: "color 0.15s, border-color 0.15s",
});

export const NavTab = ({ active, label, onClick }: NavTabProps) => (
  <button onClick={onClick} style={Style(active)}>
    {label}
  </button>
);
