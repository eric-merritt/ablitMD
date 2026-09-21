import { useState } from "react";
import { buildSomMdRecipe, type SomMdRecipe } from "../../api/ablation";

interface SomMdPanelProps {
  runId: string;
  nCategories: number;
}

// Auto-size the SOM lattice: one neuron per category minimum, square grid.
const autoGrid = (n: number): [number, number] => {
  const side = Math.max(3, Math.ceil(Math.sqrt(n)));
  return [side, side];
};

// Hexagonal SOM lattice visualization. Draws hexagons in a staggered grid
// (odd-row offset — "pointy-top" layout). Neurons are colored by activation
// (top-k get a bright fill, rest are dim).
const SomGridViz = ({
  rows,
  cols,
  k,
  bestLayer,
}: {
  rows: number;
  cols: number;
  k: number;
  bestLayer: number;
}) => {
  const totalNeurons = rows * cols;
  const pad = 4;
  const hexR = Math.max(6, Math.min(14, 220 / Math.max(rows + cols, 6)));
  const hexW = Math.sqrt(3) * hexR;
  const hexH = 2 * hexR;
  const vGap = hexH + 2;
  const hGap = hexW + 2;

  // Hexagon path (pointy-top, centered at 0,0).
  const hexPath = (() => {
    const pts: [number, number][] = [];
    for (let a = 0; a < 6; a++) {
      const angle = (Math.PI / 3) * a - Math.PI / 6;
      pts.push([hexR * Math.cos(angle), hexR * Math.sin(angle)]);
    }
    return `M${pts.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join("L")}Z`;
  })();

  const width = cols * hGap + pad * 2;
  const height = rows * vGap + pad * 2;

  // Deterministic "active" neurons: pick k positions seeded by best_layer.
  const active = new Set<number>();
  let seed = bestLayer * 7919 + k;
  for (let i = 0; i < k; i++) {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    active.add(seed % totalNeurons);
  }

  return (
    <svg width={width} height={height} style={{ display: "block" }}>
      {Array.from({ length: totalNeurons }, (_, i) => {
        const r = Math.floor(i / cols);
        const c = i % cols;
        // Odd-row offset for hexagonal stagger.
        const offsetX = (r % 2) * (hGap / 2);
        const cx = pad + c * hGap + offsetX + hexW / 2;
        const cy = pad + r * vGap + hexH / 2;
        const isActive = active.has(i);
        return (
          <g key={i} transform={`translate(${cx},${cy})`}>
            <path
              d={hexPath}
              fill={isActive ? "var(--accent)" : "var(--surface-3)"}
              stroke={isActive ? "var(--accent)" : "var(--border)"}
              strokeWidth={isActive ? 1.5 : 0.5}
              opacity={isActive ? 0.95 : 0.45}
            />
          </g>
        );
      })}
    </svg>
  );
};

export const SomMdPanel = ({ runId, nCategories }: SomMdPanelProps) => {
  const [gridOverride, setGridOverride] = useState<[number, number] | null>(null);
  const [k, setK] = useState(5);
  const [factor, setFactor] = useState(1.2);
  const [status, setStatus] = useState<"idle" | "building" | "done" | "error">("idle");
  const [recipe, setRecipe] = useState<SomMdRecipe>();
  const [error, setError] = useState<string>();

  // Grid is auto-sized from category count unless user overrides.
  const grid: [number, number] = gridOverride ?? autoGrid(nCategories);
  const maxK = grid[0] * grid[1];

  const handleBuild = async () => {
    setStatus("building");
    setError(undefined);
    try {
      const r = await buildSomMdRecipe(runId, { k, grid, factor });
      setRecipe(r);
      setStatus("done");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setStatus("error");
    }
  };

  return (
    <div
      style={{
        background: "var(--surface)",
        border: "1px solid var(--border)",
        borderRadius: "var(--radius)",
        padding: "16px",
        display: "flex",
        flexDirection: "column",
        gap: "12px",
      }}
    >
      <div style={{ fontSize: "13px", fontWeight: 700, color: "var(--text)" }}>
        SOM-MD Ablation
      </div>

      <p style={{ fontSize: "11px", color: "var(--text-muted)", margin: 0, lineHeight: 1.5 }}>
        Trains a Self-Organizing Map on harmful hidden states at the best-refusal layer,
        then ablates k directions from top neurons toward the harmless centroid across all layers.
      </p>

      <div style={{ display: "flex", alignItems: "center", gap: "16px", flexWrap: "wrap" }}>
        {/* Grid viz */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: "4px",
          }}
        >
          {recipe ? (
            <SomGridViz
              rows={recipe.grid_shape[0]}
              cols={recipe.grid_shape[1]}
              k={recipe.k}
              bestLayer={recipe.best_layer}
            />
          ) : (
            <SomGridViz rows={grid[0]} cols={grid[1]} k={k} bestLayer={0} />
          )}
          <span style={{ fontSize: "10px", color: "var(--text-dim)" }}>
            {recipe ? `${recipe.grid_shape[0]}×${recipe.grid_shape[1]}` : `${grid[0]}×${grid[1]}`}
            {!gridOverride && ` (auto: ${nCategories} cats)`}
          </span>
        </div>

        {/* Controls */}
        <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span style={{ fontSize: "11px", color: "var(--text-muted)", width: "32px" }}>k</span>
            <input
              type="number"
              min={2}
              max={maxK}
              value={k}
              onChange={(e) => setK(Math.min(maxK, Math.max(2, Number(e.target.value))))}
              style={{
                width: "48px",
                padding: "3px 5px",
                background: "var(--surface-2)",
                color: "var(--text)",
                border: "1px solid var(--border)",
                borderRadius: "4px",
                fontSize: "11px",
              }}
            />
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span style={{ fontSize: "11px", color: "var(--text-muted)", width: "32px" }}>Grid</span>
            {/* Auto-sized default */}
            <button
              onClick={() => setGridOverride(null)}
              style={{
                padding: "3px 7px",
                fontSize: "10px",
                cursor: "pointer",
                background: !gridOverride ? "var(--accent)" : "var(--surface-3)",
                color: !gridOverride ? "#fff" : "var(--text-muted)",
                border: "1px solid var(--border)",
                borderRadius: "4px",
              }}
            >
              auto ({autoGrid(nCategories)[0]}×{autoGrid(nCategories)[1]})
            </button>
            {/* Manual overrides */}
            {([[3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [10, 10]] as [number, number][]).map(
              ([r, c]) => (
                <button
                  key={`${r}x${c}`}
                  onClick={() => setGridOverride([r, c])}
                  style={{
                    padding: "3px 7px",
                    fontSize: "10px",
                    cursor: "pointer",
                    background:
                      gridOverride?.[0] === r && gridOverride?.[1] === c
                        ? "var(--accent)"
                        : "var(--surface-3)",
                    color:
                      gridOverride?.[0] === r && gridOverride?.[1] === c
                        ? "#fff"
                        : "var(--text-muted)",
                    border: "1px solid var(--border)",
                    borderRadius: "4px",
                  }}
                >
                  {r}×{c}
                </button>
              ),
            )}
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span style={{ fontSize: "11px", color: "var(--text-muted)", width: "32px" }}>×factor</span>
            <input
              type="range"
              min={0.1}
              max={3.0}
              step={0.1}
              value={factor}
              onChange={(e) => setFactor(Number(e.target.value))}
              style={{ width: "80px" }}
            />
            <span style={{ fontSize: "11px", color: "var(--text)", width: "24px" }}>
              {factor.toFixed(1)}
            </span>
          </div>

          <button
            onClick={handleBuild}
            disabled={status === "building"}
            style={{
              padding: "6px 14px",
              cursor: status === "building" ? "not-allowed" : "pointer",
              opacity: status === "building" ? 0.6 : 1,
            }}
          >
            {status === "building" ? "Training SOM…" : "Build SOM-MD Recipe"}
          </button>
        </div>
      </div>

      {/* Result */}
      {recipe && (
        <div
          style={{
            fontSize: "11px",
            color: "var(--text-muted)",
            display: "flex",
            gap: "12px",
            flexWrap: "wrap",
          }}
        >
          <span>
            Best layer: <strong style={{ color: "var(--text)" }}>{recipe.best_layer}</strong>
          </span>
          <span>
            Directions: <strong style={{ color: "var(--text)" }}>{recipe.k}</strong>
          </span>
          <span>
            Layers: <strong style={{ color: "var(--text)" }}>1–{recipe.n_layers - 1}</strong>
          </span>
          <span>
            Factor: <strong style={{ color: "var(--text)" }}>{recipe.factor}</strong>
          </span>
        </div>
      )}

      {error && (
        <div style={{ color: "#ef4444", fontSize: "12px" }}>{error}</div>
      )}
    </div>
  );
};
