import { useState } from "react";
import type { CSSProperties } from "react";

interface StartButtonProps {
  starting: boolean;
  error?: string | null;
  onClick: () => void;
}

const baseStyle: CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  alignSelf: "stretch",
  height: "360px",
  minWidth: "5rem",
  padding: ".25rem",
  cursor: "pointer",
  userSelect: "none",
  background: "var(--surface-2)",
  color: "var(--highlight)",
  borderRadius: "var(--radius)",
  transition: "color 0.15s, border-color 0.15s",
};

// Beveled ridgelines on all four edges — a raised ridge so it reads as a
// physical key, not a flat card.
const bevel = (hovered: boolean): CSSProperties => ({
  borderTop: `2px solid ${hovered ? "var(--accent)" : "var(--border-2)"}`,
  borderBottom: `2px solid ${hovered ? "var(--accent)" : "var(--border-2)"}`,
  borderLeft: `4px solid ${hovered ? "var(--accent)" : "var(--border)"}`,
  borderRight: `4px solid ${hovered ? "var(--accent)" : "var(--border)"}`,
});

// A thin vertical key on the far right of the config page. Text is horizontal,
// centered in the height; beveled edges run the full length.
const ErrorText = ({ message }: { message: string }) => (
  <span style={{ fontSize: "10px", color: "var(--text-dim)" }}>{message}</span>
);

export const StartButton = ({ starting, error, onClick }: StartButtonProps) => {
  const [hovered, setHovered] = useState(false);

  return (
    <div
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{ ...baseStyle, ...bevel(hovered), color: hovered ? "var(--accent)" : "var(--text)", fontSize: "14px", fontWeight: 600, letterSpacing: ".2rem" }}
    >
      {starting ? "…" : "START"}
      {error && <ErrorText message={error} />}
    </div>
  );
};
