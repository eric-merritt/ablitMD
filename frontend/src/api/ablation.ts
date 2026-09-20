import type {
  DivergencePayload,
  SlimRecipe,
  RecipeParams,
  VerifyEvent,
  AuditRecord,
  AuditEvent,
  AuditSummary,
  DirectionOverlapResponse,
} from "../types/ablation";

const jsonOrThrow = async (response: Response) => {
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `request failed (${response.status})`);
  }
  return response.json();
};

export const getDivergence = (runId: string): Promise<DivergencePayload> =>
  fetch(`/api/ablation/${runId}/divergence`).then(jsonOrThrow);

export const buildRecipe = (
  runId: string,
  params: RecipeParams,
): Promise<SlimRecipe> =>
  fetch(`/api/ablation/${runId}/recipe`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      onset: params.onset,
      split: params.split,
      lastLayer: params.lastLayer,
      factorA: params.factorA,
      factorB: params.factorB,
      factorAByCategory: params.factorAByCategory,
    }),
  }).then(jsonOrThrow);

export interface SomMdRecipe {
  run_id: string;
  model_id: string;
  gen_mode: string;
  method: "som_md";
  k: number;
  grid_shape: [number, number];
  best_layer: number;
  factor: number;
  n_layers: number;
  built_at: string;
}

export const buildSomMdRecipe = (
  runId: string,
  params: { k: number; grid: [number, number]; factor: number },
): Promise<SomMdRecipe> =>
  fetch(`/api/ablation/${runId}/recipe/som-md`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      k: params.k,
      grid: `${params.grid[0]},${params.grid[1]}`,
      factor: params.factor,
    }),
  }).then(jsonOrThrow);

export const bakeModel = (
  runId: string,
  mode: "ablitmd" | "classic" = "ablitmd",
  factor?: number,
  disclaimerAblate?: boolean,
  disclaimerFactor?: number,
): Promise<{ saved_to: string }> =>
  fetch(`/api/ablation/${runId}/bake`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      mode,
      ...(factor !== undefined && { factor }),
      ...(disclaimerAblate !== undefined && {
        disclaimer_ablate: disclaimerAblate,
      }),
      ...(disclaimerFactor !== undefined && {
        disclaimer_factor: disclaimerFactor,
      }),
    }),
  }).then(jsonOrThrow);

const readNdjsonStream = async <T>(
  url: string,
  body: object,
  onEvent: (event: T) => void,
  signal?: AbortSignal,
): Promise<void> => {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok || !response.body)
    throw new Error(`verify failed (${response.status})`);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (line.trim()) onEvent(JSON.parse(line));
    }
  }
};

export const verifyAblation = (
  runId: string,
  body: {
    gen_mode: string;
    categories?: string[];
    samples_per_category?: number;
  },
  onEvent: (event: VerifyEvent) => void,
  signal?: AbortSignal,
): Promise<void> =>
  readNdjsonStream(`/api/ablation/${runId}/verify`, body, onEvent, signal);

export const submitVerifyLabel = (
  label: "refused" | "complied" | "auto",
): Promise<void> =>
  fetch("/api/ablation/verify/label", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ label }),
  }).then(jsonOrThrow);

export const verifyAblationClassic = (
  runId: string,
  body: {
    gen_mode: string;
    factor: number;
    disclaimer_ablate?: boolean;
    disclaimer_factor?: number;
    categories?: string[];
    samples_per_category?: number;
  },
  onEvent: (event: VerifyEvent) => void,
  signal?: AbortSignal,
): Promise<void> =>
  readNdjsonStream(
    `/api/ablation/${runId}/verify/classic`,
    body,
    onEvent,
    signal,
  );

// --- Post-ablation adversarial audit + overlap workspace ---

// Streams the audit: watch the ablated model generate each response token-by-token,
// then see the judge's label land per trial. See AuditEvent for event shapes.
export const runAudit = (
  runId: string,
  onEvent: (event: AuditEvent) => void,
  nCategories = 5,
  rounds = 3,
  signal?: AbortSignal,
): Promise<void> =>
  readNdjsonStream<AuditEvent>(
    "/api/ablation/audit/run",
    { run_id: runId, n_categories: nCategories, rounds },
    onEvent,
    signal,
  );

export const listAudits = (runId: string): Promise<AuditSummary[]> =>
  fetch(`/api/ablation/audits?run_id=${encodeURIComponent(runId)}`).then(
    jsonOrThrow,
  );

export const directionOverlap = (body: {
  run_id: string;
  model_id: string;
  mode: string;
  experiments: { path: string; trial_indices?: number[] }[];
}): Promise<DirectionOverlapResponse> =>
  fetch("/api/ablation/direction_overlap", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then(jsonOrThrow);
