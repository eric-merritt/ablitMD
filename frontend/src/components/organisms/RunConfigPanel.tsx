import { useState, useRef } from "react";
import type { CSSProperties } from "react";
import { ModeRadioGroup } from "../molecules/ModeRadioGroup";
import { ModelCheckboxList } from "../molecules/ModelCheckboxList";
import { CategoryList } from "../molecules/CategoryList";
import { createRun, fetchRuns, fetchRun } from "../../api/runs";
import { CATEGORIES } from "../../types/categories";
import type { LLM } from "../../types/model";
import type { Run, RunSummary, RunMode } from "../../types/run";

interface RunConfigPanelProps {
  models: LLM[];
  onRunStart: (run: Run) => void;
  onRunOpen: (run: Run) => void;
}

const PanelStyle: CSSProperties = {
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
};

const ColumnCardStyle: CSSProperties = {
  flex: 1,
  background: "var(--surface-2)",
  border: "1px solid var(--border)",
  borderRadius: "var(--radius)",
  padding: "12px",
  overflow: "hidden",
};

const ColumnsStyle: CSSProperties = {
  display: "flex",
  flexDirection: "row",
  gap: "12px",
  width: "60vw",
  marginLeft: "auto",
  marginRight: "auto",
  alignItems: "flex-start",
};

const HeaderCardStyle: CSSProperties = {
  width: "60vw",
  display: "flex",
  alignItems: "center",
  gap: "10px",
  background: "var(--thirty-six)",
  border: "1px solid var(--border)",
  borderRadius: "var(--radius)",
  padding: "12px 16px",
  margin: "24px 0 12px",
};

const PrevRunsPopoverStyle: CSSProperties = {
  background: "var(--surface-2)",
  border: "1px solid var(--border)",
  borderRadius: "var(--radius)",
  display: "flex",
  flexDirection: "column",
  gap: "4px",
  padding: "8px",
  minWidth: "320px",
  maxHeight: "50vh",
  overflowY: "auto",
};

const PagesIcon = () => (
  <svg
    width="13"
    height="13"
    viewBox="0 0 13 13"
    fill="none"
    style={{ display: "inline-block", verticalAlign: "-2px", marginRight: "5px" }}
  >
    <rect x="2.5" y="0.5" width="8" height="10" rx="1" stroke="currentColor" strokeWidth="1.1" />
    <rect x="0.5" y="2.5" width="8" height="10" rx="1" stroke="currentColor" strokeWidth="1.1" fill="var(--surface)" />
  </svg>
);

const missingLabel = (noModels: boolean, noCategories: boolean) => {
  if (noModels && noCategories)
    return "Select at least one model and one category";
  if (noModels) return "Select at least one model";
  return "Select at least one category";
};

export const RunConfigPanel = ({
  models,
  onRunStart,
  onRunOpen,
}: RunConfigPanelProps) => {
  const [mode, setMode] = useState("non_thinking");
  const [selectedModels, setSelectedModels] = useState<Set<string>>(new Set());
  const [selectedCategories, setSelectedCategories] = useState<Set<string>>(
    new Set(CATEGORIES.map((cat) => cat.id)),
  );
  const [existingRuns, setExistingRuns] = useState<RunSummary[] | null>(null);
  const [loadingRuns, setLoadingRuns] = useState(false);
  const [starting, setStarting] = useState(false);
  const [startHovered, setStartHovered] = useState(false);
  const [prevRunsHovered, setPrevRunsHovered] = useState(false);
  const [startError, setStartError] = useState<string | null>(null);
  const [tooltipVisible, setTooltipVisible] = useState(false);
  const errorTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const canStart = selectedModels.size > 0 && selectedCategories.size > 0;

  const handleModelToggle = (modelId: string, checked: boolean) =>
    setSelectedModels((prev) => {
      const next = new Set(prev);
      checked ? next.add(modelId) : next.delete(modelId);
      return next;
    });

  const handleCategoryToggle = (categoryId: string, checked: boolean) =>
    setSelectedCategories((prev) => {
      const next = new Set(prev);
      checked ? next.add(categoryId) : next.delete(categoryId);
      return next;
    });

  const handleGroupToggle = (groupId: string, checked: boolean) => {
    const groupCatIds = CATEGORIES.filter((cat) => cat.group === groupId).map(
      (cat) => cat.id,
    );
    setSelectedCategories((prev) => {
      const next = new Set(prev);
      groupCatIds.forEach((id) => (checked ? next.add(id) : next.delete(id)));
      return next;
    });
  };

  const handleStart = async () => {
    if (starting) return;
    if (!canStart) {
      const msg = missingLabel(
        selectedModels.size === 0,
        selectedCategories.size === 0,
      );
      setStartError(msg);
      if (errorTimer.current) clearTimeout(errorTimer.current);
      errorTimer.current = setTimeout(() => setStartError(null), 3000);
      return;
    }
    setStarting(true);
    try {
      const run = await createRun({
        models: [...selectedModels],
        mode_selection: mode as RunMode,
        prompt_scope: { categories: [...selectedCategories] },
      });
      onRunStart(run);
    } finally {
      setStarting(false);
    }
  };

  const handleOpenExisting = async () => {
    if (existingRuns) {
      setExistingRuns(null);
      return;
    }
    setLoadingRuns(true);
    try {
      const runs = await fetchRuns();
      setExistingRuns(runs);
    } finally {
      setLoadingRuns(false);
    }
  };

  const handleSelectRun = async (summary: RunSummary) => {
    const run = await fetchRun(summary.run_id);
    onRunOpen(run);
  };

  return (
    <div style={{ position: "relative", height: "100%", width: "100%" }}>
      <div style={PanelStyle}>
        <div style={HeaderCardStyle}>
          <span
            style={{
              fontSize: "28px",
              fontWeight: 600,
              letterSpacing: "0.08em",
              color: "var(--text-dim)",
            }}
          >
            Configuration
          </span>
          <div
            style={{ position: "relative", display: "inline-flex" }}
            onMouseEnter={() => setTooltipVisible(true)}
            onMouseLeave={() => setTooltipVisible(false)}
          >
            <span
              style={{
                fontSize: "12px",
                color: "var(--text-muted)",
                cursor: "default",
                userSelect: "none",
              }}
            >
              ⓘ
            </span>
            {tooltipVisible && (
              <div
                style={{
                  position: "absolute",
                  left: "20px",
                  top: "-4px",
                  background: "var(--surface)",
                  border: "1px solid var(--border)",
                  borderRadius: "var(--radius)",
                  padding: "6px 10px",
                  fontSize: "12px",
                  color: "var(--text-dim)",
                  whiteSpace: "nowrap",
                  zIndex: 20,
                  pointerEvents: "none",
                }}
              >
                Select models, categories, and run mode before starting.
              </div>
            )}
          </div>
        </div>
        <div style={ColumnsStyle}>
          <div style={{ ...ColumnCardStyle, flexShrink: 1 }}>
            <ModeRadioGroup selected={mode} onChange={setMode} />
          </div>
          <div style={{ ...ColumnCardStyle, flex: 2, minWidth: "160px" }}>
            <ModelCheckboxList
              models={models}
              selectedModels={selectedModels}
              onModelToggle={handleModelToggle}
            />
          </div>
          <div
            style={{
              ...ColumnCardStyle,
              flex: 2,
              minWidth: "220px",
              height: "676px",
              overflowY: "auto",
            }}
          >
            <CategoryList
              selectedCategories={selectedCategories}
              onCategoryToggle={handleCategoryToggle}
              onGroupToggle={handleGroupToggle}
            />
          </div>
        </div>
      </div>

      <div
        style={{
          position: "absolute",
          bottom: 0,
          left: 0,
          right: 0,
          zIndex: 10,
        }}
      >
        <hr style={{ border: "none", borderTop: "1px solid var(--border)", margin: 0 }} />
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            padding: "10px 32px",
          }}
        >
          <div style={{ position: "relative" }}>
            <span
              onClick={handleOpenExisting}
              onMouseEnter={() => setPrevRunsHovered(true)}
              onMouseLeave={() => setPrevRunsHovered(false)}
              style={{
                color: prevRunsHovered ? "var(--accent)" : "var(--text)",
                fontSize: "19px",
                cursor: "pointer",
                userSelect: "none",
                transition: "color 0.15s",
              }}
            >
              <PagesIcon />
              <span style={{ textDecoration: prevRunsHovered ? "underline" : "none" }}>
                {loadingRuns ? "Loading…" : "Previous Runs"}
              </span>
            </span>
            {existingRuns && (
              <div style={{ ...PrevRunsPopoverStyle, position: "absolute", bottom: "calc(100% + 8px)", left: 0 }}>
                {existingRuns.length === 0 && (
                  <div style={{ color: "var(--text-muted)", padding: "4px", fontSize: "12px" }}>
                    No runs yet.
                  </div>
                )}
                {existingRuns.map((run) => (
                  <ul
                    key={run.run_id}
                    onClick={() => handleSelectRun(run)}
                    style={{
                      padding: "8px 10px",
                      background: "var(--surface-3)",
                      border: "1px solid var(--border)",
                      borderRadius: "var(--radius)",
                      cursor: "pointer",
                      listStyle: "none",
                    }}
                  >
                    <li
                      style={{
                        color: "var(--text-dim)",
                        fontSize: "12px",
                        display: "flex",
                        justifyContent: "space-between",
                      }}
                    >
                      <span>{new Date(run.started_at).toLocaleString()}</span>
                      <span style={{ color: run.incomplete ? "#eab308" : "#10b981" }}>
                        {run.incomplete ? "Incomplete" : "Complete"}
                      </span>
                    </li>
                    <li>
                      <ul
                        style={{
                          paddingLeft: "16px",
                          marginTop: "4px",
                          listStyle: "disc",
                          color: "var(--text-dim)",
                          fontSize: "11px",
                          display: "flex",
                          flexDirection: "column",
                          gap: "2px",
                        }}
                      >
                        <li>Mode: {run.mode_selection ?? "—"}</li>
                        <li>Models: {(run.models ?? []).join(", ") || "—"}</li>
                        <li>{run.prompt_count ?? 0} prompts</li>
                      </ul>
                    </li>
                  </ul>
                ))}
              </div>
            )}
          </div>

          <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: "6px" }}>
            {startError && (
              <div
                style={{
                  border: "1px solid #ef4444",
                  borderRadius: "var(--radius)",
                  padding: "5px 10px",
                  fontSize: "12px",
                  color: "#ef4444",
                  background: "var(--surface)",
                  animation: "fadeIn 0.15s ease",
                }}
              >
                {startError}
              </div>
            )}
            <span
              onClick={handleStart}
              onMouseEnter={() => setStartHovered(true)}
              onMouseLeave={() => setStartHovered(false)}
              style={{
                color: startHovered ? "var(--accent)" : "var(--text)",
                fontSize: "19px",
                cursor: "pointer",
                userSelect: "none",
                transition: "color 0.15s",
              }}
            >
              {starting ? (
                "Starting…"
              ) : (
                <>
                  <span style={{ textDecoration: startHovered ? "underline" : "none" }}>
                    Start
                  </span>{" "}
                  →
                </>
              )}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
