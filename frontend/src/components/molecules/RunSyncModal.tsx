import { useState } from "react";
import type { CSSProperties } from "react";

interface RunSyncModalProps {
  runIds: string[];
  syncing: boolean;
  onConfirm: (selectedRunIds: string[]) => void;
  onCancel: () => void;
}

interface RowPartProps {
  runId: string;
  checked: boolean;
  onToggle: (runId: string, checked: boolean) => void;
}

interface RunChecklistProps {
  runIds: string[];
  selected: Set<string>;
  onToggle: (runId: string, checked: boolean) => void;
}

interface ModalActionsProps {
  confirmDisabled: boolean;
  syncing: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

interface ModalCardProps extends RunChecklistProps, ModalActionsProps {}

const OverlayStyle: CSSProperties = {
  position: "fixed",
  inset: 0,
  background: "rgba(0, 0, 0, 0.55)",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  zIndex: 100,
};

const CardStyle: CSSProperties = {
  background: "var(--surface-2)",
  border: "1px solid var(--border)",
  borderRadius: "var(--radius)",
  padding: "16px",
  minWidth: "420px",
  maxWidth: "560px",
  maxHeight: "70vh",
  display: "flex",
  flexDirection: "column",
  gap: "12px",
};

const ChecklistStyle: CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: "4px",
  overflowY: "auto",
  listStyle: "none",
  margin: 0,
  padding: 0,
};

const RowLabelStyle: CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: "10px",
  padding: "8px 10px",
  background: "var(--surface-3)",
  border: "1px solid var(--border)",
  borderRadius: "var(--radius)",
  cursor: "pointer",
  fontSize: "13px",
  color: "var(--text-dim)",
};

const buttonStyle = (disabled: boolean, primary: boolean): CSSProperties => ({
  padding: primary ? "6px 18px" : "6px 14px",
  background: primary && !disabled ? "var(--accent)" : "var(--surface-3)",
  border: "1px solid var(--border)",
  borderRadius: "var(--radius)",
  color: primary ? (disabled ? "var(--text-muted)" : "var(--surface)") : "var(--text)",
  cursor: disabled ? "default" : "pointer",
});

const createEmptySelection = () => new Set<string>();

// run_2026-05-17T19-04-50-053Z_e219fff8 -> human date + short hash
const parseRunId = (runId: string) => {
  const match = runId.match(/^run_(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})-(\d{2})-(\d{3})Z_(.+)$/);
  if (!match) return { dateLabel: runId, hashLabel: "" };
  const [, datePart, hours, minutes, seconds, millis, hash] = match;
  const startedAt = new Date(`${datePart}T${hours}:${minutes}:${seconds}.${millis}Z`);
  return { dateLabel: startedAt.toLocaleString(), hashLabel: hash };
};

const ModalTitle = () => (
  <span style={{ fontSize: "15px", fontWeight: 600, color: "var(--text-dim)" }}>
    Choose which runs you'd like to sync:
  </span>
);

const RunCheckbox = ({ runId, checked, onToggle }: RowPartProps) => (
  <input
    type="checkbox"
    checked={checked}
    onChange={(event) => onToggle(runId, event.target.checked)}
  />
);

const RunDateLabel = ({ runId }: { runId: string }) => (
  <span style={{ flex: 1 }}>{parseRunId(runId).dateLabel}</span>
);

const RunHashLabel = ({ runId }: { runId: string }) => (
  <span style={{ color: "var(--text-muted)", fontSize: "11px" }}>
    {parseRunId(runId).hashLabel}
  </span>
);

const RunRowLabel = ({ runId, checked, onToggle }: RowPartProps) => (
  <label style={RowLabelStyle}>
    <RunCheckbox runId={runId} checked={checked} onToggle={onToggle} />
    <RunDateLabel runId={runId} />
    <RunHashLabel runId={runId} />
  </label>
);

const RunChecklistRow = ({ runId, checked, onToggle }: RowPartProps) => (
  <li>
    <RunRowLabel runId={runId} checked={checked} onToggle={onToggle} />
  </li>
);

const RunChecklist = ({ runIds, selected, onToggle }: RunChecklistProps) => (
  <ul style={ChecklistStyle}>
    {runIds.map((runId) => (
      <RunChecklistRow
        key={runId}
        runId={runId}
        checked={selected.has(runId)}
        onToggle={onToggle}
      />
    ))}
  </ul>
);

const CancelButton = ({ syncing, onCancel }: Pick<ModalActionsProps, "syncing" | "onCancel">) => (
  <button onClick={onCancel} disabled={syncing} style={buttonStyle(syncing, false)}>
    Cancel
  </button>
);

const OkButton = ({ confirmDisabled, syncing, onConfirm }: Omit<ModalActionsProps, "onCancel">) => (
  <button onClick={onConfirm} disabled={confirmDisabled} style={buttonStyle(confirmDisabled, true)}>
    {syncing ? "Syncing…" : "Ok"}
  </button>
);

const ModalActions = ({ confirmDisabled, syncing, onConfirm, onCancel }: ModalActionsProps) => (
  <div style={{ display: "flex", justifyContent: "flex-end", gap: "10px" }}>
    <CancelButton syncing={syncing} onCancel={onCancel} />
    <OkButton confirmDisabled={confirmDisabled} syncing={syncing} onConfirm={onConfirm} />
  </div>
);

const ModalCard = (props: ModalCardProps) => (
  <div style={CardStyle}>
    <ModalTitle />
    <RunChecklist runIds={props.runIds} selected={props.selected} onToggle={props.onToggle} />
    <ModalActions
      confirmDisabled={props.confirmDisabled}
      syncing={props.syncing}
      onConfirm={props.onConfirm}
      onCancel={props.onCancel}
    />
  </div>
);

export const RunSyncModal = ({ runIds, syncing, onConfirm, onCancel }: RunSyncModalProps) => {
  const [selected, setSelected] = useState(createEmptySelection());

  const handleToggle = (runId: string, checked: boolean) =>
    setSelected((prev) => {
      const next = new Set(prev);
      checked ? next.add(runId) : next.delete(runId);
      return next;
    });

  return (
    <div style={OverlayStyle}>
      <ModalCard
        runIds={runIds}
        selected={selected}
        onToggle={handleToggle}
        confirmDisabled={selected.size === 0 || syncing}
        syncing={syncing}
        onConfirm={() => onConfirm([...selected])}
        onCancel={onCancel}
      />
    </div>
  );
};
