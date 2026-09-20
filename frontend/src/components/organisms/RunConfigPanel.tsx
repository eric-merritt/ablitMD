import { useState, useRef, useEffect } from "react";
import type { CSSProperties } from "react";
import { NavBarTabbed } from "../molecules/NavBarTabbed";
import { PageFooter } from "../molecules/PageFooter";
import { RunCard } from "../molecules/RunCard";
import { ModelCard } from "../molecules/ModelCard";
import { StartButton } from "../molecules/StartButton";
import { ModeRadioGroup } from "../molecules/ModeRadioGroup";
import { CategoryList } from "../molecules/CategoryList";
import { RunSyncModal } from "../molecules/RunSyncModal";
import { createRun, fetchRuns, fetchRun, fetchRemoteRunIds, syncRuns } from "../../api/runs";
import { CATEGORIES } from "../../types/categories";
import type { LLM } from "../../types/model";
import type { Run, RunSummary, RunMode } from "../../types/run";

interface RunConfigPanelProps {
  models: LLM[];
  onRunStart: (run: Run) => void;
  onRunOpen: (run: Run) => void;
}

type Tab = "new" | "partial" | "completed";

const TABS: { id: Tab; label: string }[] = [
  { id: "new", label: "New Run" },
  { id: "partial", label: "Continue Partial Run" },
  { id: "completed", label: "Completed Runs" },
];

const PanelStyle: CSSProperties = {
  display: "flex",
  flexDirection: "row",
  alignItems: "center",
};

const ColumnCardStyle: CSSProperties = {
  flex: 1,
  background: "var(--surface-2)",
  border: "1px solid var(--border)",
  borderRadius: "var(--radius)",
  overflow: "scroll",
  width: "360px",
  height: "360px",
  padding: "1.5rem",
};

const ColumnsStyle: CSSProperties = {
  display: "flex",
  flexDirection: "row",
  gap: "12px",
  width: "fit-content",
  marginLeft: "auto",
  marginRight: "auto",
  alignItems: "flex-start",
};

const TabContentStyle: CSSProperties = {
  padding: "16px 32px 24px",
  overflowY: "auto",
};

const missingLabel = (noModels: boolean, noCategories: boolean) => {
  if (noModels && noCategories) return "Select at least one model and one category";
  if (noModels) return "Select at least one model";
  return "Select at least one category";
};

export const RunConfigPanel = ({ models, onRunStart, onRunOpen }: RunConfigPanelProps) => {
  const modelNames: Record<string, string> = {};
  for (const m of models) modelNames[m.modelId] = m.name;

  const [tab, setTab] = useState<Tab>("new");
  const [mode, setMode] = useState("non_thinking");
  const [selectedCard, setSelectedCard] = useState<string | null>(null);
  const [selectedCategories, setSelectedCategories] = useState<Set<string>>(
    new Set(CATEGORIES.map((cat) => cat.id)),
  );
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [loadingRuns, setLoadingRuns] = useState(false);
  const [starting, setStarting] = useState(false);
  const [remoteRunIds, setRemoteRunIds] = useState<string[] | null>(null);
  const [loadingRemote, setLoadingRemote] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [mergeError, setMergeError] = useState<string | null>(null);
  const [startError, setStartError] = useState<string | null>(null);
  const errorTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const canStart = selectedCard !== null && selectedCategories.size > 0;

  // Load the run list once — both the partial and completed tabs read from it.
  useEffect(() => {
    let cancelled = false;
    setLoadingRuns(true);
    fetchRuns()
      .then((r) => { if (!cancelled) setRuns(r) })
      .catch(() => {})
      .finally(() => { if (!cancelled) setLoadingRuns(false) });
    return () => { cancelled = true };
  }, []);

  // Single-select, owned here so only one card can be marked at a time. A second
  // click on the same card does not unset it — clicking always selects.
  const selectModel = (modelId: string) => setSelectedCard(modelId);

  const handleCategoryToggle = (categoryId: string, checked: boolean) =>
    setSelectedCategories((prev) => {
      const next = new Set(prev);
      checked ? next.add(categoryId) : next.delete(categoryId);
      return next;
    });

  const handleGroupToggle = (groupId: string, checked: boolean) => {
    const groupCatIds = CATEGORIES.filter((cat) => cat.group === groupId).map((cat) => cat.id);
    setSelectedCategories((prev) => {
      const next = new Set(prev);
      groupCatIds.forEach((id) => (checked ? next.add(id) : next.delete(id)));
      return next;
    });
  };

  const handleStart = async () => {
    if (starting) return;
    if (!canStart) {
      const msg = missingLabel(selectedCard === null, selectedCategories.size === 0);
      setStartError(msg);
      if (errorTimer.current) clearTimeout(errorTimer.current);
      errorTimer.current = setTimeout(() => setStartError(null), 3000);
      return;
    }
    setStarting(true);
    try {
      const run = await createRun({
        models: [selectedCard],
        mode_selection: mode as RunMode,
        prompt_scope: { categories: [...selectedCategories] },
      });
      onRunStart(run);
    } finally {
      setStarting(false);
    }
  };

  // Resume a run at its current step (partial) or open it to results (complete).
  const handleSelectRun = async (summary: RunSummary) => {
    const run = await fetchRun(summary.run_id);
    onRunOpen(run);
  };

  const handleMergeOpen = async () => {
    if (loadingRemote || remoteRunIds) return;
    setLoadingRemote(true);
    setMergeError(null);
    try {
      setRemoteRunIds(await fetchRemoteRunIds());
    } catch {
      setMergeError("Remote data server unreachable");
      setTimeout(() => setMergeError(null), 3000);
    } finally {
      setLoadingRemote(false);
    }
  };

  const handleSyncConfirm = async (selectedRunIds: string[]) => {
    setSyncing(true);
    try {
      await syncRuns(selectedRunIds);
      setRemoteRunIds(null);
      setRuns(await fetchRuns());
    } catch {
      setMergeError("Sync failed — see backend log");
      setTimeout(() => setMergeError(null), 3000);
    } finally {
      setSyncing(false);
    }
  };

  const partialRuns = (runs ?? []).filter((r) => r.incomplete);
  const completedRuns = (runs ?? []).filter((r) => !r.incomplete);

  return (
    <NavBarTabbed tabs={TABS} active={tab} onSelect={(id) => setTab(id as Tab)}>
      <div style={{ ...PanelStyle, alignItems: "center", border: "1px solid var(--border)", gap: "0px", margin: "0 1.5rem 1.5rem 1.5rem", borderRadius: "var(--radius)", position: "relative", top: "calc(1.5rem - 1px)", zIndex: 0, height: "100%", padding: "1.5rem" }}>
        {tab === "new" && (
          <div style={ColumnsStyle}>
            <div style={{ ...ColumnCardStyle, flexShrink: 1 }}>
              <ModeRadioGroup selected={mode} onChange={setMode} />
            </div>
            <div style={{ ...ColumnCardStyle, flex: 2, minWidth: "160px", display: "flex", flexDirection: "column", gap: "8px" }}>
              <div style={{ fontSize: "13px", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em", background: "var(--surface-3)", border: "1px solid var(--border-2)", borderRadius: "var(--radius)", padding: "7px 10px", color: "var(--text-muted)" }}>
                Models
              </div>
              {models.map((model) => (
                <ModelCard
                  key={model.modelId}
                  label={model.name}
                  selected={selectedCard === model.modelId}
                  onClick={() => selectModel(model.modelId)}
                />
              ))}
            </div>
            <div style={{ ...ColumnCardStyle, flex: 2, minWidth: "220px", maxHeight: "25%", overflowY: "auto" }}>
              <CategoryList
                selectedCategories={selectedCategories}
                onCategoryToggle={handleCategoryToggle}
                onGroupToggle={handleGroupToggle}
              />
            </div>
            <StartButton starting={starting} error={startError} onClick={handleStart} />
          </div>
        )}

        {tab === "partial" && (
          <div style={{ ...TabContentStyle, overflow: "scroll", height: "100%", width: "100%" }}>
            {loadingRuns && <div style={{ fontSize: "13px", color: "var(--text-muted)" }}>Loading…</div>}
            {!loadingRuns && partialRuns.length === 0 && (
              <div style={{ fontSize: "13px", color: "var(--text-muted)" }}>No partial runs.</div>
            )}
            <div style={{ display: "flex", flexWrap: "wrap", margin: "1rem", gap: "0.5rem" }}>
              {partialRuns.map((run) => (
                <RunCard key={run.run_id} summary={run} onOpen={handleSelectRun} modelNames={modelNames} />
              ))}
            </div>
          </div>
        )}

        {tab === "completed" && (
          <div style={TabContentStyle}>
            {loadingRuns && <div style={{ fontSize: "13px", color: "var(--text-muted)" }}>Loading…</div>}
            {!loadingRuns && completedRuns.length === 0 && (
              <div style={{ fontSize: "13px", color: "var(--text-muted)" }}>No completed runs yet.</div>
            )}
            <div style={{ display: "grid", gridTemplateColumns: "minmax(280px, 340px) 1fr", gap: "24px" }}>
              <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
                {completedRuns.map((run) => (
                  <RunCard key={run.run_id} summary={run} onOpen={handleSelectRun} modelNames={modelNames} />
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      {remoteRunIds && (
        <RunSyncModal
          runIds={remoteRunIds}
          syncing={syncing}
          onConfirm={handleSyncConfirm}
          onCancel={() => setRemoteRunIds(null)}
        />
      )}

      <PageFooter
        onMerge={handleMergeOpen}
        merging={loadingRemote}
        mergeError={mergeError}
      />
    </NavBarTabbed>
  );
};
