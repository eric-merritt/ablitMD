import { useState } from "react";
import type { CSSProperties } from "react";

interface RunModeChoiceProps {
  onManual: () => void;
  onAudit: () => void;
}

const ChoiceStyle: CSSProperties = {
  flex: 1,
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  justifyContent: "center",
  gap: "8px",
  padding: "24px 32px",
  background: "var(--surface-2)",
  color: "var(--text)",
  border: "2px solid var(--border)",
  borderRadius: "var(--radius)",
  cursor: "pointer",
  userSelect: "none",
  transition: "border-color 0.15s, color 0.15s",
};

const Choice = ({ label, sub, onPick }: { label: string; sub: string; onPick: () => void }) => {
  const [hovered, setHovered] = useState(false);
  return (
    <div
      onClick={onPick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{ ...ChoiceStyle, borderColor: hovered ? "var(--accent-dim)" : undefined }}
    >
      <span style={{ fontSize: "18px", fontWeight: 600 }}>{label}</span>
      <span style={{ fontSize: "13px", color: "var(--text-dim)" }}>{sub}</span>
    </div>
  );
};

// Shown when a completed run is opened: pick how to abliterate it.
export const RunModeChoice = ({ onManual, onAudit }: RunModeChoiceProps) => (
  <div style={{ display: "flex", gap: "16px", width: "fit-content", margin: "0 auto" }}>
    <Choice label="Manual Abliteration" sub="Abliterate the directions yourself" onPick={onManual} />
    <Choice label="Agent Audit" sub="Let the agent audit the run" onPick={onAudit} />
  </div>
);
