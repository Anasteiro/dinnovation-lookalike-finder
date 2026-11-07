import { afterEach, describe, expect, it, vi } from "vitest";
import { acceptsFile, api, API_BASE, percent, subscribeJob } from "./api";

afterEach(() => vi.unstubAllGlobals());
describe("Local API contract", () => {
  it("accepts xlsx and csv, but no other extension", () => {
    expect(acceptsFile("clients.xlsx")).toBe(true);
    expect(acceptsFile("clients.CSV")).toBe(true);
    expect(acceptsFile("clients.xls")).toBe(false);
    expect(acceptsFile("clients.xlsx.exe")).toBe(false);
  });
  it("uses localhost:8000 by default", () => {
    expect(API_BASE).toBe("http://localhost:8000");
  });
  it("posts use_sample=true without a file", async () => {
    const fetcher = vi.fn().mockResolvedValue(new Response('{"job_id":"job-1"}'));
    vi.stubGlobal("fetch", fetcher);
    expect(await api.analyze(null)).toEqual({ job_id: "job-1" });
    const call = fetcher.mock.calls[0];
    if (!call) throw new Error("Expected analyze request");
    const [url, init] = call;
    expect(url).toBe("http://localhost:8000/api/analyze");
    expect(init.method).toBe("POST");
    expect(init.body.get("use_sample")).toBe("true");
    expect(init.body.has("file")).toBe(false);
  });
  it("posts the actual selected file, not its name or sample data", async () => {
    const fetcher = vi.fn().mockResolvedValue(new Response('{"job_id":"file-job"}'));
    vi.stubGlobal("fetch", fetcher);
    const file = new File(["vat\n123"], "clients.csv", { type: "text/csv" });
    await api.analyze(file);
    const call = fetcher.mock.calls[0];
    if (!call) throw new Error("Expected file request");
    const body = call[1].body as FormData;
    expect(body.get("file")).toBe(file);
    expect(body.has("use_sample")).toBe(false);
  });
  it("sends pagination, profile, and search to the server", async () => {
    const fetcher = vi.fn().mockResolvedValue(new Response('{"total":0,"items":[]}'));
    vi.stubGlobal("fetch", fetcher);
    await api.lookalikes("job-1", "icp-1", 20, 40, "Αθήνα & ΣΙΑ");
    const call = fetcher.mock.calls[0];
    if (!call) throw new Error("Expected lookalike request");
    const params = new URL(call[0]).searchParams;
    expect(Object.fromEntries(params)).toEqual({ job: "job-1", profile: "icp-1", limit: "20", offset: "40", q: "Αθήνα & ΣΙΑ" });
  });
  it("preserves server detail text for 400 responses", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response('{"detail":"Invalid VAT column"}', { status: 400 })));
    await expect(api.analyze(null)).rejects.toThrow("Invalid VAT column");
  });
  it("preserves plain text server errors", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("Market file missing", { status: 500 })));
    await expect(api.health()).rejects.toThrow("Market file missing");
  });
  it("handles empty and unreachable responses without fabricated results", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("")));
    await expect(api.results("job-1")).rejects.toThrow("κενή απάντηση");
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    await expect(api.health()).rejects.toThrow("Δεν βρέθηκε ο τοπικός server στο http://localhost:8000");
  });
  it("formats API fractions as percentages", () => {
    expect(percent(0.81)).toBe("81%");
    expect(percent(0)).toBe("0%");
    expect(percent(1)).toBe("100%");
  });
  it("generates all four exact export endpoint names", () => {
    for (const name of ["profiles.json", "assignments.csv", "lookalikes.csv", "lookalikes_global.csv"] as const) {
      expect(api.exportUrl("job-1", name)).toBe(`http://localhost:8000/api/export/job-1/${name}`);
    }
  });
});

describe("Real SSE events", () => {
  function setup() {
    let source: Stream | undefined;
    class Stream {
      onmessage?: (message: { data: string }) => void;
      onerror?: () => void;
      end?: (event: { data: string }) => void;
      close = vi.fn();
      constructor(public url: string) { source = this; }
      addEventListener(name: string, callback: (event: { data: string }) => void) { if (name === "end") this.end = callback; }
    }
    vi.stubGlobal("EventSource", Stream);
    const handlers = { event: vi.fn(), end: vi.fn(), error: vi.fn() };
    const cleanup = subscribeJob("job-1", handlers);
    if (!source) throw new Error("EventSource was not created");
    return { source, handlers, cleanup };
  }
  it("passes real elapsed times and matched counts without a timer", () => {
    const { source, handlers } = setup();
    const event = { step: "matched", t: 2.7, matched_vats: 7, client_vats: 9 };
    source.onmessage?.({ data: JSON.stringify(event) });
    expect(source.url).toBe("http://localhost:8000/api/jobs/job-1/events");
    expect(handlers.event).toHaveBeenCalledWith(event);
    expect(handlers.end).not.toHaveBeenCalled();
  });
  it("waits for event:end status done rather than a step:done message", () => {
    const { source, handlers } = setup();
    source.onmessage?.({ data: '{"step":"done","t":3}' });
    expect(handlers.end).not.toHaveBeenCalled();
    source.end?.({ data: '{"status":"done"}' });
    expect(handlers.end).toHaveBeenCalledWith("done");
    expect(source.close).toHaveBeenCalled();
  });
  it("stops and preserves the server's analysis error", () => {
    const { source, handlers } = setup();
    source.onmessage?.({ data: '{"step":"error","t":1,"message":"No matching VATs"}' });
    expect(handlers.error).toHaveBeenCalledWith("No matching VATs");
    expect(source.close).toHaveBeenCalled();
    source.end?.({ data: '{"status":"done"}' });
    expect(handlers.end).not.toHaveBeenCalled();
  });
});