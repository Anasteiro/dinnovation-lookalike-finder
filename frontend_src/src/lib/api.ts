export const API_BASE = (import.meta.env['VITE_API_BASE'] || "http://localhost:8000").replace(/\/+$/, "");

export type Health = { status: string; data_found: boolean; market_data_loaded: boolean; ollama_running: boolean; llm_model: string; embedding: string };
export type Feature = { label: string; support: number; share_in_profile: number; share_in_universe: number; lift: number; shrunk_lift: number; p_value: number; q_value: number };
export type Profile = {
  profile_id: string; title: string; description: string; description_is_ai: boolean;
  description_meta: { ai_label: string; source: string; verified: boolean; model: string; paragraphs?: Paragraph[] };
  n_clients: number; turnover_share: number | null; stability: number; confidence: number; features: Feature[];
  activity_clarity?: ClaritySummary | null;
};
export type ClaritySummary = { profile_mean: number; market_mean: number; profile_noisy_share: number; market_noisy_share: number };
export type ClarityKad = { code: string; type: string; description: string; related: boolean | null };
export type ClarityCompany = { company_id: string; clarity: number; level: string; declared_other_kads: number; unrelated_kads: number; explanation: string; kads: ClarityKad[]; kads_total: number };
export type ClarityOverview = { method: string; levels: Record<string, number>; histogram: { edges: number[]; counts: number[] }; market_mean: number; profiles: Record<string, ClaritySummary | null>; note: string };
export type Dataset = {
  market_size: number; candidates: number; client_vats: number; matched_vats: number; unmatched_vats: number;
  client_shops: number; profiles_found: number; lookalike_rows: number; none_share: number;
  execution_seconds: number; source: string;
};
export type AnalysisResult = { job_id: string; dataset: Dataset; meta: Record<string, unknown>; profiles: Profile[] };
export type Lookalike = { tier?: string; candidates?: number; rank: number; company_id: string | number; profile_id: string; score: number; reason: string; name: string; municipality: string | null; prefecture: string | null; legal_type: string | null; primary_kad_label: string | null; founded: string | number | null; lat: number | null; lon: number | null; clarity?: number | null; clarity_level?: string | null; declared_other_kads?: number | null; unrelated_kads?: number | null };
export type LookalikePage = { total: number; items: Lookalike[] };
export type JobEvent = { t: number; step: string; message?: string; detail?: string; error?: string; [key: string]: unknown };
export type ExportFile = "profiles.json" | "assignments.csv" | "lookalikes.csv" | "lookalikes_global.csv" | "lookalikes_by_profile.csv" | "activity_clarity.csv" | "refined_lookalikes.csv";
export type Paragraph = { title?: string; text: string; is_ai?: boolean };
export type Narration = { status: "idle" | "running" | "done" | "error"; done: number; total: number; message?: string };
export type JobStatus = { narration?: Narration; [key: string]: unknown };
export type BenchmarkMethod = {
  label: string;
  "recall@500": number;
  "recall@2000": number;
  "recall@500_ci95"?: [number, number];
  "recall@2000_ci95"?: [number, number];
};
export type BenchmarkResult = {
  methods: { baseline: BenchmarkMethod; knn: BenchmarkMethod; product: BenchmarkMethod };
  lift_vs_baseline: { "500": number; "2000": number };
  random_level: number | { "500": number; "2000": number };
  paired_wins: number; repeats: number; n_hidden_per_split: number; note: string;
};
export type BenchmarkStatus = Partial<BenchmarkResult> & {
  status: "idle" | "running" | "done" | "error";
  message?: string; error?: string;
  result?: BenchmarkResult;
};
export type GeoKind = "market" | "clients" | "lookalikes";
export type GeoCell = { lat: number; lon: number; n: number };
export type GeoDensity = { cell_deg: number; total: number; bounds: { lat: [number, number]; lon: [number, number] }; cells: GeoCell[] };
export type Municipality = { municipality: string; lookalikes: number; market: number; share_of_market_in_top: number; lat: number; lon: number };
export type GeoOutline = { type: "Feature"; geometry: { type: "MultiPolygon"; coordinates: number[][][][] }; properties?: Record<string, unknown> };
export type EnrichmentStatus = { available?: boolean; [key: string]: unknown };
export type Place = { name: string; rating: number | null; reviews: number | null; lat: number; lon: number; [key: string]: unknown };
export type Multipliers = { activity: number; geo: number; legal: number; age: number };
export type WhatIfEvaluation = Pick<BenchmarkResult, "methods" | "lift_vs_baseline" | "repeats" | "note">;
export type WhatIfResult = { multipliers: Multipliers; items: { rank: number; company_id: string | number; name: string; municipality: string; default_rank: number | null; moved: number }[]; overlap_with_default: { "500": number; "2000": number }; is_default: boolean; evaluation?: WhatIfEvaluation };
export type TransparencyProfile = { profile_id?: string; title?: string; stability?: number; confidence?: number; description_is_ai?: boolean; description_meta?: { verified?: boolean; source?: string; facts?: { id?: string; text?: string }[] } };
export type Transparency = {
  execution?: { external_ai_calls?: number; timings_s?: Record<string, number> };
  models?: { embedding?: string; llm?: string; llm_options?: Record<string, unknown>; clustering?: string; statistics?: string };
  selection?: { rule?: string; k_table?: Record<string, unknown>[] };
  profiles?: TransparencyProfile[];
  licenses?: Record<string, unknown>[];
  ai_act?: Record<string, unknown>[];
  limits?: string[];
};

export class ApiUnavailableError extends Error {
  constructor() { super(`Δεν βρέθηκε ο τοπικός server στο ${API_BASE}`); }
}

async function responseError(response: Response): Promise<Error> {
  const text = await response.text();
  try {
    const body = JSON.parse(text) as { detail?: unknown; message?: unknown };
    const detail = body.detail ?? body.message;
    return new Error(typeof detail === "string" ? detail : detail ? JSON.stringify(detail) : text || response.statusText);
  } catch { return new Error(text || response.statusText || `HTTP ${response.status}`); }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try { response = await fetch(`${API_BASE}${path}`, init); }
  catch (error) {
    if (error instanceof Error && error.name === "AbortError") throw error;
    throw new ApiUnavailableError();
  }
  if (!response.ok) throw await responseError(response);
  const text = await response.text();
  if (!text.trim()) throw new Error("Ο server επέστρεψε κενή απάντηση.");
  return JSON.parse(text) as T;
}

let outlineCache: Promise<GeoOutline> | null = null;

export const acceptsFile = (name: string) => /\.(xlsx|csv)$/i.test(name);
export const percent = (value: number) => `${(value * 100).toLocaleString("el-GR", { maximumFractionDigits: 1 })}%`;
export const fmtQ = (q: number) => q === 0 ? "< 1e-300" : q < 0.001 ? q.toExponential(1).replace(".", ",") : q.toLocaleString("el-GR", { maximumSignificantDigits: 2 });
export type FactorInfo = { key: string; label: string; unit: string; source: string; higher: string; coverage: number; min: number | null; max: number | null; market_median: number | null };
export type ProfileFactor = { key: string; label: string; unit: string; profile_median: number; market_median: number; p_value: number; significant: boolean };
export type FactorsResponse = { factors: FactorInfo[]; options: { legal: { value: string; count: number }[]; prefectures: string[]; founded: [number, number] | null; branches: number }; profiles: { profile_id: string; title: string; external_factors: ProfileFactor[] }[]; note: string };
export type RefineFilters = { legal?: string[]; prefectures?: string[]; min_clarity?: number; founded_from?: number; founded_to?: number; exclude_branches?: boolean; factors?: Record<string, { min?: number | null; max?: number | null }> };
export type RefineRequest = { profile: string; filters: RefineFilters; boosts: Record<string, number>; strength: number; limit: number; offset: number };
export type RefineItem = Lookalike & { model_rank: number; refined_score: number; [factor: string]: unknown };
export type RefineResponse = { total: number; removed: Record<string, number>; offset: number; limit: number; items: RefineItem[]; top2000_medians: Record<string, number | null>; export: string };
export type ChatRow = Record<string, string | number | null>;
export type ChatAnswer = { question: string; text: string; kind: "answer" | "info" | "help" | "refusal" | "fallback"; rows: ChatRow[]; columns: string[]; sources: string[]; filters: string; suggestions?: string[]; ai_generated: boolean; ai_summary: { text: string | null; label: string } | null; llm_available: boolean; seconds: number };
export const CHAT_MAX = 400;
export const api = {
  chat: (job: string, question: string, aiSummary = false, signal?: AbortSignal) => request<ChatAnswer>(`/api/jobs/${encodeURIComponent(job)}/chat`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: question.slice(0, CHAT_MAX), ai_summary: aiSummary }), signal: signal ?? null }),
  factors: (job: string, signal?: AbortSignal) => request<FactorsResponse>(`/api/factors?job=${encodeURIComponent(job)}`, { signal: signal ?? null }),
  refine: (job: string, body: RefineRequest, signal?: AbortSignal) => request<RefineResponse>(`/api/jobs/${encodeURIComponent(job)}/refine`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal: signal ?? null }),
  clarityOverview: (job: string, signal?: AbortSignal) => request<ClarityOverview>(`/api/clarity?job=${encodeURIComponent(job)}`, { signal: signal ?? null }),
  clarityCompany: (job: string, companyId: string | number, signal?: AbortSignal) => request<ClarityCompany>(`/api/clarity?job=${encodeURIComponent(job)}&company_id=${encodeURIComponent(String(companyId))}`, { signal: signal ?? null }),
  health: (signal?: AbortSignal) => request<Health>("/api/health", { signal: signal ?? null }),
  analyze: (file: File | null) => {
    const body = new FormData();
    if (file) body.append("file", file); else body.append("use_sample", "true");
    return request<{ job_id: string }>("/api/analyze", { method: "POST", body });
  },
  results: (job: string, signal?: AbortSignal) => request<AnalysisResult>(`/api/results/${encodeURIComponent(job)}`, { signal: signal ?? null }),
  jobStatus: (job: string, signal?: AbortSignal) => request<JobStatus>(`/api/jobs/${encodeURIComponent(job)}`, { signal: signal ?? null }),
  benchmarkStatus: (job: string, signal?: AbortSignal) => request<BenchmarkStatus>(`/api/jobs/${encodeURIComponent(job)}/benchmark`, { signal: signal ?? null }),
  benchmark: async (job: string, signal?: AbortSignal) => {
    const body = new FormData(); body.append("repeats", "8");
    let response: Response;
    try { response = await fetch(`${API_BASE}/api/jobs/${encodeURIComponent(job)}/benchmark`, { method: "POST", body, signal: signal ?? null }); }
    catch (error) { if (error instanceof Error && error.name === "AbortError") throw error; throw new ApiUnavailableError(); }
    if (!response.ok) throw await responseError(response);
  },
  narrate: async (job: string) => {
    let response: Response;
    try { response = await fetch(`${API_BASE}/api/jobs/${encodeURIComponent(job)}/narrate`, { method: "POST" }); }
    catch { throw new ApiUnavailableError(); }
    if (!response.ok) throw await responseError(response);
  },
  transparency: (job: string, signal?: AbortSignal) => request<Transparency>(`/api/transparency/${encodeURIComponent(job)}`, { signal: signal ?? null }),
  density: (job: string, kind: GeoKind, profile: string, cell: number, signal?: AbortSignal) => {
    const params = new URLSearchParams({ job, kind, profile, cell: String(cell) });
    return request<GeoDensity>(`/api/geo/density?${params}`, { signal: signal ?? null });
  },
  municipalities: (job: string, profile: string, signal?: AbortSignal) => {
    const params = new URLSearchParams({ job, profile, limit: "12" });
    return request<{ items: Municipality[] }>(`/api/geo/municipalities?${params}`, { signal: signal ?? null });
  },
  outline: (signal?: AbortSignal) => {
    outlineCache ??= request<GeoOutline>("/api/geo/outline").catch(error => { outlineCache = null; throw error; });
    if (!signal) return outlineCache;
    return Promise.race([outlineCache, new Promise<never>((_, reject) => {
      if (signal.aborted) reject(new DOMException("Aborted", "AbortError"));
      else signal.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")), { once: true });
    })]);
  },
  enrichmentStatus: (signal?: AbortSignal) => request<EnrichmentStatus>("/api/enrichment/status", { signal: signal ?? null }),
  places: async (job: string, limit = 300, signal?: AbortSignal) => {
    const params = new URLSearchParams({ job, limit: String(limit) });
    const body = await request<{ items?: Place[] } | Place[]>(`/api/enrichment/places?${params}`, { signal: signal ?? null });
    return Array.isArray(body) ? body : (body.items ?? []);
  },
  whatif: (job: string, multipliers: Multipliers, evaluate = false, signal?: AbortSignal) => {
    const body = new FormData();
    for (const [key, value] of Object.entries(multipliers)) body.append(key, String(value));
    body.append("top", "25");
    if (evaluate) body.append("evaluate", "true");
    return request<WhatIfResult>(`/api/jobs/${encodeURIComponent(job)}/whatif`, { method: "POST", body, signal: signal ?? null });
  },
  lookalikes: (job: string, profile: string, limit: number, offset: number, q: string, signal?: AbortSignal) => {
    const params = new URLSearchParams({ job, profile, limit: String(limit), offset: String(offset), q });
    return request<LookalikePage>(`/api/lookalikes?${params}`, { signal: signal ?? null });
  },
  exportUrl: (job: string, file: ExportFile) => `${API_BASE}/api/export/${encodeURIComponent(job)}/${file}`,
  download: async (job: string, file: ExportFile) => {
    let response: Response;
    try { response = await fetch(api.exportUrl(job, file)); } catch { throw new ApiUnavailableError(); }
    if (!response.ok) throw await responseError(response);
    const blob = await response.blob();
    if (!blob.size) throw new Error("Το αρχείο εξαγωγής είναι κενό.");
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url; link.download = file; link.click();
    URL.revokeObjectURL(url);
  },
};

export function subscribeJob(job: string, handlers: { event: (event: JobEvent) => void; end: (status: string) => void; error: (message: string) => void }): () => void {
  const url = `${API_BASE}/api/jobs/${encodeURIComponent(job)}/events`;
  const source = new EventSource(url);
  let closed = false;
  const close = () => { closed = true; source.close(); };
  source.onmessage = (message) => {
    if (closed) return;
    try {
      const event = JSON.parse(message.data) as JobEvent;
      handlers.event(event);
      if (event.step === "error") { close(); handlers.error(event.message ?? event.detail ?? event.error ?? "Η ανάλυση απέτυχε."); }
    } catch { close(); handlers.error("Μη έγκυρο συμβάν από τον server."); }
  };
  source.addEventListener("end", (message) => {
    if (closed) return;
    close();
    try { handlers.end((JSON.parse((message as MessageEvent).data) as { status: string }).status); }
    catch { handlers.error("Μη έγκυρη τελική απάντηση από τον server."); }
  });
  source.onerror = async () => {
    if (closed) return;
    source.close();
    const controller = new AbortController();
    try {
      const response = await fetch(url, { signal: controller.signal });
      if (!closed) handlers.error(response.ok ? "Διακόπηκε η σύνδεση με την ανάλυση." : (await responseError(response)).message);
    } catch { if (!closed) handlers.error(new ApiUnavailableError().message); }
    finally { controller.abort(); close(); }
  };
  return close;
}
