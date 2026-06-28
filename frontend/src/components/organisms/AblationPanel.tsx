import { useEffect, useState, useRef } from "react";
import { DivergenceChart } from "../molecules/DivergenceChart";
import { LayerSplitControls } from "../molecules/LayerSplitControls";
import { CategoryFactorSliders } from "../molecules/CategoryFactorSliders";
import { RecipeReuseWarning } from "../molecules/RecipeReuseWarning";
import { RecipePanel } from "../molecules/RecipePanel";
import { getDivergence, buildRecipe, bakeModel } from "../../api/ablation";
import type {
  ModeDivergence,
  SlimRecipe,
  RecipeParams,
} from "../../types/ablation";

type AblationMode = "ablitmd" | "classic";

interface AblationPanelProps {
  runId: string;
  models: { modelId: string; apiModelId: string; name: string }[];
  onVerify: (
    genMode: string,
    mode: AblationMode,
    classicFactor: number,
    disclaimerAblate: boolean,
    disclaimerFactor: number,
  ) => void;
}

type ModelLoadState = "idle" | "loading" | "ready" | "error";

const GatedButton = ({
  disabled,
  tooltip,
  onClick,
  children,
}: {
  disabled: boolean;
  tooltip: string;
  onClick: () => void;
  children: React.ReactNode;
}) => {
  const [hovered, setHovered] = useState(false);
  return (
    <span
      style={{ position: "relative", display: "inline-block" }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <button
        onClick={onClick}
        disabled={disabled}
        style={{
          padding: "6px 14px",
          cursor: disabled ? "not-allowed" : "pointer",
        }}
      >
        {children}
      </button>
      {disabled && hovered && tooltip && (
        <span
          style={{
            position: "absolute",
            bottom: "calc(100% + 6px)",
            left: "50%",
            transform: "translateX(-50%)",
            background: "var(--surface-3)",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius)",
            padding: "4px 8px",
            fontSize: "11px",
            color: "var(--text)",
            whiteSpace: "nowrap",
            pointerEvents: "none",
            zIndex: 10,
          }}
        >
          {tooltip}
        </span>
      )}
    </span>
  );
};

const SliderWithInput = ({
  min,
  max,
  step,
  value,
  onChange,
}: {
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (n: number) => void;
}) => {
  const [draft, setDraft] = useState(value.toFixed(2));
  useEffect(() => {
    setDraft(value.toFixed(2));
  }, [value]);

  const commit = () => {
    const next = Number(draft);
    if (Number.isFinite(next) && next >= min && next <= max) onChange(next);
    else setDraft(value.toFixed(2));
  };

  return (
    <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ width: "80px" }}
      />
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") commit();
        }}
        style={{
          width: "58px",
          padding: "3px 5px",
          background: "var(--surface-2)",
          color: "var(--text)",
          border: "1px solid var(--border)",
          borderRadius: "4px",
          fontSize: "11px",
        }}
      />
    </div>
  );
};

const pickFirstModelEntry = (payload: Record<string, unknown>) => {
  for (const [key, value] of Object.entries(payload)) {
    if (key !== "computed_at" && value && typeof value === "object") {
      const genModes = value as Record<string, Record<string, ModeDivergence>>;
      const firstGenMode = Object.values(genModes)[0];
      if (firstGenMode) return { modelId: key, modeMap: firstGenMode };
    }
  }
  return null;
};

const DEFAULT_PARAMS: RecipeParams = {
  onset: 38,
  split: 50,
  factorA: 0.15,
  factorB: 0.15,
  factorAByCategory: {},
};

export const AblationPanel = ({
  runId,
  models,
  onVerify,
}: AblationPanelProps) => {
  const [hard, setHard] = useState<ModeDivergence>();
  const [redirect, setRedirect] = useState<ModeDivergence>();
  const [none, setNone] = useState<ModeDivergence>();
  const [params, setParams] = useState<RecipeParams>(DEFAULT_PARAMS);
  const [recipe, setRecipe] = useState<SlimRecipe>();
  const [status, setStatus] = useState<
    "idle" | "loading" | "building" | "applied"
  >("loading");
  const [error, setError] = useState<string>();
  const [bakedPath, setBakedPath] = useState<string>();
  const [baking, setBaking] = useState(false);
  const [modelId, setModelId] = useState<string>();
  const [modelLoad, setModelLoad] = useState<ModelLoadState>("idle");
  const [mode, setMode] = useState<AblationMode>("ablitmd");
  const [classicFactor, setClassicFactor] = useState(0.6);
  const [disclaimerAblate, setDisclaimerAblate] = useState(false);
  const [disclaimerFactor, setDisclaimerFactor] = useState(0.3);
  const [builtParams, setBuiltParams] = useState<RecipeParams | null>(null);
  const buildRequestIdRef = useRef(0);

  useEffect(() => {
    getDivergence(runId)
      .then(async (payload) => {
        const entry = pickFirstModelEntry(payload as Record<string, unknown>);
        if (!entry) {
          setError("No divergence data for this run");
          setStatus("idle");
          return;
        }
        const { modelId: foundModelId, modeMap } = entry;
        setModelId(foundModelId);
        setHard(modeMap.hard);
        setRedirect(modeMap.redirect);
        setNone(modeMap.none);
        const reference = modeMap.hard ?? modeMap.redirect;
        const initialParams = reference
          ? {
              ...DEFAULT_PARAMS,
              onset: reference.suggested_onset,
              split:
                reference.suggested_split ?? reference.suggested_divergence,
            }
          : DEFAULT_PARAMS;
        setParams(initialParams);
        setStatus("building");
        const requestId = ++buildRequestIdRef.current;
        buildRecipe(runId, initialParams)
          .then((r) => {
            if (requestId === buildRequestIdRef.current) {
              setRecipe(r);
              setBuiltParams(initialParams);
            }
            setStatus("idle");
          })
          .catch((err) => {
            if (requestId === buildRequestIdRef.current) {
              setError(String(err.message));
            }
            setStatus("idle");
          });
      })
      .catch((err) => {
        setError(String(err.message));
        setStatus("idle");
      });
  }, [runId, models]);

  useEffect(() => {
    if (!builtParams) return;
    const same = JSON.stringify(params) === JSON.stringify(builtParams);
    if (same) return;
    setStatus("building");
    const requestId = ++buildRequestIdRef.current;
    const timer = setTimeout(() => {
      buildRecipe(runId, params)
        // A superseded request (newer knob change in flight) owns the state now —
        // it must not write recipe/status/error, or it would re-enable the action
        // buttons while a stale recipe is still the newest file on disk.
        .then((next) => {
          if (requestId !== buildRequestIdRef.current) return;
          setRecipe(next);
          setBuiltParams(params);
          setStatus("idle");
        })
        .catch((err) => {
          if (requestId !== buildRequestIdRef.current) return;
          setError(String(err.message));
          setStatus("idle");
        });
    }, 400);
    return () => clearTimeout(timer);
  }, [params, builtParams]);

  const runBake = () => {
    setBaking(true);
    setError(undefined);
    bakeModel(runId)
      .then((result) => setBakedPath(result.saved_to))
      .catch((err) => setError(String(err.message)))
      .finally(() => setBaking(false));
  };

  // Master factorA broadcasts to every category: moving it clears all per-category
  // overrides, re-syncing them to the master value. Other params leave overrides intact.
  const handleParamsChange = (next: RecipeParams) => {
    if (next.factorA !== params.factorA) {
      setParams({ ...next, factorAByCategory: {} });
    } else {
      setParams(next);
    }
  };

  const handleCategoryFactor = (category: string, value: number) => {
    setParams({
      ...params,
      factorAByCategory: { ...params.factorAByCategory, [category]: value },
    });
  };

  const categories = recipe
    ? Object.values(recipe.modes)[0]?.phase_a.category_ids ?? []
    : [];

  const lastLayer = (hard ?? redirect)?.clumping.length
    ? (hard ?? redirect)!.clumping.length - 1
    : params.split;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
      <div style={{ fontSize: "15px", fontWeight: 700, color: "var(--text)" }}>
        Abliteration Pipeline
      </div>

      {error && (
        <div style={{ color: "#ef4444", fontSize: "12px" }}>{error}</div>
      )}
      {status === "loading" && (
        <div style={{ color: "var(--text-muted)", fontSize: "12px" }}>
          Loading divergence…
        </div>
      )}

      {(hard || redirect) && (
        <>
          <DivergenceChart
            hard={hard}
            redirect={redirect}
            none={none}
            onset={params.onset}
            split={params.split}
            lastLayer={lastLayer}
            factorA={params.factorA}
            factorB={params.factorB}
          />
          <div
            style={{
              display: "flex",
              gap: "16px",
              alignItems: "flex-end",
              flexWrap: "wrap",
            }}
          >
            <LayerSplitControls
              params={params}
              lastLayer={lastLayer}
              onChange={handleParamsChange}
            />
            <div style={{ display: "flex", gap: "4px", alignItems: "center" }}>
              {(["ablitmd", "classic"] as AblationMode[]).map((m) => (
                <button
                  key={m}
                  onClick={() => setMode(m)}
                  style={{
                    padding: "4px 10px",
                    fontSize: "11px",
                    cursor: "pointer",
                    background:
                      mode === m
                        ? m === "classic"
                          ? "#1e40af"
                          : "var(--accent)"
                        : "var(--surface-3)",
                    color: mode === m ? "#fff" : "var(--text-muted)",
                    border: "1px solid var(--border)",
                    borderRadius: "var(--radius)",
                    fontWeight: mode === m ? 700 : 400,
                  }}
                >
                  {m === "ablitmd" ? "ablitMD" : "Classic"}
                </button>
              ))}
            </div>
            {mode === "classic" && (
              <div
                style={{ display: "flex", alignItems: "center", gap: "6px" }}
              >
                <span style={{ fontSize: "11px", color: "var(--text-muted)" }}>
                  ×
                </span>
                <SliderWithInput
                  min={0.1}
                  max={3.0}
                  step={0.05}
                  value={classicFactor}
                  onChange={setClassicFactor}
                />
              </div>
            )}
            <GatedButton
              disabled={!recipe || status === "building"}
              tooltip={status === "building" ? "Recipe rebuilding…" : ""}
              onClick={() => {
                if (recipe) {
                  onVerify(
                    recipe.gen_mode,
                    mode,
                    classicFactor,
                    disclaimerAblate,
                    disclaimerFactor,
                  );
                }
              }}
            >
              Verify
            </GatedButton>
            <GatedButton
              disabled={!recipe || baking || status === "building"}
              tooltip={status === "building" ? "Recipe rebuilding…" : ""}
              onClick={runBake}
            >
              {baking ? "Baking…" : "Bake & save model"}
            </GatedButton>
          </div>
          {mode === "ablitmd" && recipe?.prior_attempt && (
            <RecipeReuseWarning
              priorAttempt={recipe.prior_attempt}
              onset={recipe.onset}
              split={recipe.split}
              factorA={recipe.factor_a}
              factorB={recipe.factor_b}
            />
          )}
          {mode === "ablitmd" && (
            <CategoryFactorSliders
              categories={categories}
              factorAByCategory={params.factorAByCategory}
              masterFactor={params.factorA}
              onChange={handleCategoryFactor}
            />
          )}
          {status === "building" && (
            <div style={{ color: "var(--text-muted)", fontSize: "11px" }}>
              Building recipe…
            </div>
          )}

          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "12px",
              paddingTop: "8px",
              borderTop: "1px solid var(--border)",
            }}
          >
            <label
              style={{
                display: "flex",
                alignItems: "center",
                gap: "6px",
                cursor: "pointer",
              }}
            >
              <input
                type="checkbox"
                checked={disclaimerAblate}
                onChange={(e) => setDisclaimerAblate(e.target.checked)}
                style={{ cursor: "pointer" }}
              />
              <span
                style={{
                  fontSize: "11px",
                  color: "var(--text)",
                  textTransform: "uppercase",
                  letterSpacing: "0.04em",
                }}
              >
                Disclaimer ablation
              </span>
            </label>
            {disclaimerAblate && (
              <div
                style={{ display: "flex", alignItems: "center", gap: "6px" }}
              >
                <span style={{ fontSize: "11px", color: "var(--text-muted)" }}>
                  ×
                </span>
                <SliderWithInput
                  min={0.05}
                  max={1.0}
                  step={0.05}
                  value={disclaimerFactor}
                  onChange={setDisclaimerFactor}
                />
              </div>
            )}
          </div>
        </>
      )}

      {recipe && (
        <>
          <RecipePanel
            recipe={recipe}
            onset={params.onset}
            split={params.split}
          />

          {bakedPath && (
            <div style={{ color: "var(--accent)", fontSize: "12px" }}>
              Saved abliterated model to <code>{bakedPath}</code>
            </div>
          )}
        </>
      )}
    </div>
  );
};
