import type { CSSProperties } from "react";
import type { RunSummary } from "../../types/run";

interface RunCardProps {
  summary: RunSummary;
  onOpen: (summary: RunSummary) => void;
  modelNames: Record<string, string>;
}

const CardStyle: CSSProperties = {
  display: "flex",
  fontSize: "18px",
  flexDirection: "column",
  gap: "1rem",
  background: "var(--surface-2)",
  border: "1px solid var(--border)",
  borderRadius: "var(--radius)",
  padding: "16px 16px 16px 32px",
  width: "28vw",
  margin: "0 auto",
};

const MetaStyle: CSSProperties = {
  fontSize: "16px",
  color: "var(--text-dim)",
  display: "flex",
  flexDirection: "column",
  paddingLeft: ".5rem",
};

// Accent when the run is complete, dimmed while it's still in progress.
const StatusBadge = ({ incomplete }: { incomplete: boolean }) => (
  <span
    style={{
      fontSize: "12px",
      fontWeight: 600,
      color: incomplete ? "var(--text-dim)" : "var(--accent)",
      background: "var(--surface-3)",
      border: `1px solid ${incomplete ? "var(--border-2)" : "var(--accent)"}`,
      borderRadius: "var(--radius)",
      padding: "2px 8px",
    }}
  >
    {incomplete ? "In Progress" : "Complete"}
  </span>
);

const modelTitle = (models: string[], modelNames: Record<string, string>) =>
  models.map((id) => {
    const name = modelNames[id] ?? id;
    return name.split("Base")[0].split(" ").join("-").slice(0, -1);
  });

// One run card. Clicking it opens the run — partial runs resume from where they
// left off, completed runs open to results. No confirmation either way.
export const RunCard = ({ summary, onOpen, modelNames }: RunCardProps) => (
  <a onClick={() => onOpen(summary)}>
    <div style={CardStyle}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <span style={{ fontSize: "24px", letterSpacing: ".25rem", fontWeight: 700, color: "var(--text)" }}>
          {modelTitle(summary.models, modelNames)}
        </span>
        <StatusBadge incomplete={summary.incomplete} />
      </div>

      <div style={MetaStyle}>
        <div>Mode: {summary.mode_selection ?? "—"}</div>
        <div>Prompt Count: {summary.prompt_count ?? 0}</div>
      </div>

      <div style={{ display: "flex", justifyContent: "flex-end" }}>
        <span style={{ fontSize: "14px", color: "var(--text-dim)" }}>
          {new Date(summary.started_at).toLocaleString()}
        </span>
      </div>
    </div>
  </a>
);
