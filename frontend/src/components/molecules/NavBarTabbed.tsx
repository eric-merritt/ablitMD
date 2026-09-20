import type { CSSProperties, ReactNode } from "react";
import { NavTab } from "./NavTab";

interface NavBarTabbedProps {
  tabs: { id: string; label: string }[];
  active: string;
  onSelect: (id: string) => void;
  children: ReactNode;
}

const StripStyle: CSSProperties = {
  display: "flex",
  width: "fit-content",
  margin: "1rem auto",
  padding: "0 8px",
};

// Manila-folder tab strip: the full-width bottom border is the section's top edge.
// The active tab paints over it (no border of its own) so it reads as sitting on top;
// inactive tabs keep their bottom border and sit a hair lower.
export const NavBarTabbed = ({ tabs, active, onSelect, children }: NavBarTabbedProps) => (
  <div style={{ position: "relative", zIndex: 15, flex: 1, overflowY: "auto", display: "flex", flexDirection: "column", alignItems: "stretch" }}>
    <div style={{ ...StripStyle, margin: "auto", marginTop: "1.5rem", zIndex: 15, position: "relative", top: "1.5rem" }}>
      {tabs.map((t) => (
        <NavTab key={t.id} active={active === t.id} label={t.label} onClick={() => onSelect(t.id)} />
      ))}
    </div>
    {children}
  </div>
);
