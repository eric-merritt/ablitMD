import type { CSSProperties } from "react";
import { Button } from "../atoms/Button";

// TODO: loading state — show a spinner or disabled state while merging

interface PageFooterProps {
  onMerge: () => void;
  merging: boolean;
  mergeError: string | null;
}

const FooterStyle: CSSProperties = {
  position: "sticky",
  bottom: 0,
  left: 0,
  right: 0,
  background: "var(--surface)",
  borderTop: "1px solid var(--border)",
  padding: "10px 32px",
  marginTop: "1.5rem",
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
};

const MergeError = ({ message }: { message: string }) => (
  <span style={{ color: "var(--text-dim)", fontSize: "12px" }}>{message}</span>
);

export const PageFooter = ({ onMerge, merging, mergeError }: PageFooterProps) => (
  <div style={FooterStyle}>
    <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-start", gap: "6px" }}>
      <Button label={merging ? "Merging…" : "Merge Run Data"} onClick={onMerge} />
      {mergeError && <MergeError message={mergeError} />}
    </div>
  </div>
);
