import { afterEach, describe, expect, it, vi } from "vitest";
import { api, type WhatIfResult } from "./api";
import { DEFAULT_MULTIPLIERS, densityOpacity, geoProjection, scheduleWhatIf, WEIGHT_RANGE } from "./exploration";

afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); });
const response = () => new Response(JSON.stringify({ multipliers: DEFAULT_MULTIPLIERS, items: [], overlap_with_default: { "500": 500, "2000": 2000 }, is_default: true }));
describe("Local exploration contracts", () => {
  it("uses default weights of 1 and the requested 0–2 range in steps of 0.1", () => {
    expect(DEFAULT_MULTIPLIERS).toEqual({ activity: 1, geo: 1, legal: 1, age: 1 });
    expect(WEIGHT_RANGE).toEqual({ min: 0, max: 2, step: 0.1 });
  });
  it("sends all density filters to the local API", async () => {
    const fetcher = vi.fn().mockResolvedValue(new Response('{}')); vi.stubGlobal("fetch", fetcher);
    await api.density("job/1", "lookalikes", "icp-1", 0.1);
    const url = new URL(fetcher.mock.calls[0]?.[0]);
    expect(url.origin + url.pathname).toBe("http://localhost:8000/api/geo/density");
    expect(Object.fromEntries(url.searchParams)).toEqual({ job: "job/1", kind: "lookalikes", profile: "icp-1", cell: "0.1" });
  });
  it("requests twelve municipalities for the selected profile", async () => {
    const fetcher = vi.fn().mockResolvedValue(new Response('{"items":[]}')); vi.stubGlobal("fetch", fetcher);
    await api.municipalities("job-1", "icp-2");
    const url = new URL(fetcher.mock.calls[0]?.[0]);
    expect(url.pathname).toBe("/api/geo/municipalities");
    expect(Object.fromEntries(url.searchParams)).toEqual({ job: "job-1", profile: "icp-2", limit: "12" });
  });
  it("posts four multipart weights and exactly top=25", async () => {
    const fetcher = vi.fn().mockResolvedValue(response()); vi.stubGlobal("fetch", fetcher);
    await api.whatif("job/1", { activity: 0, geo: 2, legal: 1.1, age: 1 });
    const call = fetcher.mock.calls[0]; if (!call) throw new Error("Missing request");
    expect(call[0]).toBe("http://localhost:8000/api/jobs/job%2F1/whatif");
    expect(call[1].method).toBe("POST");
    expect(Array.from((call[1].body as FormData).entries())).toEqual([["activity", "0"], ["geo", "2"], ["legal", "1.1"], ["age", "1"], ["top", "25"]]);
  });
  it("evaluates on the same endpoint with evaluate=true", async () => {
    const fetcher = vi.fn().mockResolvedValue(response()); vi.stubGlobal("fetch", fetcher);
    await api.whatif("job-1", DEFAULT_MULTIPLIERS, true);
    expect(fetcher.mock.calls[0]?.[0]).toBe("http://localhost:8000/api/jobs/job-1/whatif");
    expect(fetcher.mock.calls[0]?.[1].body.get("evaluate")).toBe("true");
  });
  it("waits 600 ms before computing a changed ranking", async () => {
    vi.useFakeTimers(); const fetcher = vi.fn().mockResolvedValue(response()); vi.stubGlobal("fetch", fetcher);
    const stop = scheduleWhatIf("job-1", DEFAULT_MULTIPLIERS, false, { result: vi.fn(), error: vi.fn(), started: vi.fn() });
    await vi.advanceTimersByTimeAsync(599); expect(fetcher).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1); expect(fetcher).toHaveBeenCalledTimes(1); stop();
  });
  it("cancels the previous timer and aborts an in-flight request without applying stale results", async () => {
    vi.useFakeTimers(); let resolve: ((value: Response) => void) | undefined;
    const fetcher = vi.fn((_url: string, _init?: RequestInit) => new Promise<Response>(r => { resolve = r; })); vi.stubGlobal("fetch", fetcher);
    const handlers = { result: vi.fn<(value: WhatIfResult) => void>(), error: vi.fn(), started: vi.fn() };
    const cancelTimer = scheduleWhatIf("job-1", DEFAULT_MULTIPLIERS, false, handlers); cancelTimer();
    await vi.advanceTimersByTimeAsync(600); expect(fetcher).not.toHaveBeenCalled();
    const cancelRequest = scheduleWhatIf("job-1", DEFAULT_MULTIPLIERS, false, handlers);
    await vi.advanceTimersByTimeAsync(600); cancelRequest(); resolve?.(response());
    await vi.advanceTimersByTimeAsync(0);
    expect(fetcher.mock.calls[0]?.[1]?.signal?.aborted).toBe(true);
    expect(handlers.result).not.toHaveBeenCalled(); expect(handlers.error).not.toHaveBeenCalled();
  });
  it("projects longitude with cos(38°) and latitude downwards", () => {
    const projection = geoProjection({ cell_deg: 0.1, total: 1, bounds: { lat: [37, 39], lon: [22, 24] }, cells: [] });
    expect(projection.x(24) - projection.x(23)).toBeCloseTo(Math.cos(38 * Math.PI / 180));
    expect(projection.y(38) - projection.y(39)).toBeCloseTo(1);
    expect(projection.cellWidth / projection.cellHeight).toBeCloseTo(Math.cos(38 * Math.PI / 180));
  });
  it("scales cell opacity from 0.15 to 0.95 with square-root density", () => {
    expect(densityOpacity(0, 100)).toBe(0.15);
    expect(densityOpacity(100, 100)).toBeCloseTo(0.95);
    expect(densityOpacity(25, 100)).toBeCloseTo(0.55);
  });
});