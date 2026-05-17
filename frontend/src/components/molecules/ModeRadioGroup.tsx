import { ModeRadio } from "../atoms/ModeRadio";

interface ModeRadioGroupProps {
  selected: string;
  onChange: (mode: string) => void;
}

const MODES = [
  { value: "non_thinking", label: "Non-Thinking" },
  { value: "thinking", label: "Thinking" },
  { value: "both", label: "Both" },
];

const SectionLabel = () => (
  <div style={{
    fontSize: "13px",
    fontWeight: 600,
    textTransform: "uppercase",
    letterSpacing: "0.08em",
    marginBottom: "10px",
    background: "var(--surface-3)",
    border: "1px solid var(--border-2)",
    borderRadius: "var(--radius)",
    padding: "7px 10px",
    color: "var(--text-muted)",
  }}>
    Mode
  </div>
);

export const ModeRadioGroup = ({ selected, onChange }: ModeRadioGroupProps) => (
  <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
    <li><SectionLabel /></li>
    <li>
      <ul style={{ paddingLeft: "16px", listStyle: "none", display: "flex", flexDirection: "column", gap: "4px" }}>
        {MODES.map((m) => (
          <li key={m.value}>
            <ModeRadio value={m.value} label={m.label} selected={selected} onChange={onChange} />
          </li>
        ))}
      </ul>
    </li>
  </ul>
);
