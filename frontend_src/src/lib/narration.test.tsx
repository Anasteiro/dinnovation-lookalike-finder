import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { AI_BADGE, AUTO_BADGE, paragraphBadge, profileParagraphs, watchNarration } from "./narration";
import type { Profile } from "./api";
import { ProfileCard } from "../components/forcecrm/Profiles";

afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); });

const baseProfile: Profile = {
  profile_id: "icp-1", title: "Δοκιμαστικό προφίλ", description: "Περιγραφή χωρίς παραγράφους.", description_is_ai: false,
  description_meta: { ai_label: "EuroLLM τοπικά", source: "stats", verified: true, model: "eurollm" },
  n_clients: 12, turnover_share: null, stability: 0.8, confidence: 0.7, features: [],
};

describe("Paragraph badges", () => {
  it("labels AI paragraphs as local model and the rest as automatic", () => {
    expect(paragraphBadge({ text: "x", is_ai: true })).toEqual({ label: AI_BADGE, ai: true });
    expect(paragraphBadge({ text: "x", is_ai: false })).toEqual({ label: AUTO_BADGE, ai: false });
    expect(paragraphBadge({ text: "x" }).ai).toBe(false);
  });
  it("uses paragraphs only when present and non-empty", () => {
    expect(profileParagraphs(baseProfile)).toBeNull();
    expect(profileParagraphs({ ...baseProfile, description_meta: { ...baseProfile.description_meta, paragraphs: [] } })).toBeNull();
    const paragraphs = [{ title: "A", text: "B", is_ai: true }];
    expect(profileParagraphs({ ...baseProfile, description_meta: { ...baseProfile.description_meta, paragraphs } })).toBe(paragraphs);
  });
  it("renders one badge per paragraph with the right labels", () => {
    render(<ProfileCard p={{ ...baseProfile, description_meta: { ...baseProfile.description_meta, paragraphs: [
      { title: "Ποιοι είναι", text: "Κείμενο από το μοντέλο.", is_ai: true },
      { title: "Αριθμοί", text: "Κείμενο από στατιστικά.", is_ai: false },
    ] } }} onViewLookalikes={() => {}} />);
    expect(screen.getByText(AI_BADGE)).toBeInTheDocument();
    expect(screen.getByText(AUTO_BADGE)).toBeInTheDocument();
    expect(screen.getByText("EuroLLM τοπικά")).toBeInTheDocument();
  });
  it("falls back to the single description when paragraphs are missing", () => {
    render(<ProfileCard p={baseProfile} onViewLookalikes={() => {}} />);
    expect(screen.getByText("Περιγραφή χωρίς παραγράφους.")).toBeInTheDocument();
    expect(screen.getByText("Αυτόματο κείμενο από στατιστικά")).toBeInTheDocument();
    expect(screen.queryByText(AUTO_BADGE)).not.toBeInTheDocument();
  });
});

describe("Narration polling", () => {
  const job = (narration: unknown) => new Response(JSON.stringify({ narration }));
  it("polls every 1.5s while running and stops on done", async () => {
    vi.useFakeTimers();
    const fetcher = vi.fn()
      .mockResolvedValueOnce(job({ status: "running", done: 1, total: 3 }))
      .mockResolvedValueOnce(job({ status: "running", done: 2, total: 3 }))
      .mockResolvedValueOnce(job({ status: "done", done: 3, total: 3 }));
    vi.stubGlobal("fetch", fetcher);
    const handlers = { update: vi.fn(), done: vi.fn(), error: vi.fn() };
    const stop = watchNarration("job-1", handlers);
    await vi.advanceTimersByTimeAsync(0);
    expect(fetcher).toHaveBeenCalledTimes(1);
    expect(fetcher.mock.calls[0]?.[0]).toBe("http://localhost:8000/api/jobs/job-1");
    await vi.advanceTimersByTimeAsync(1500);
    expect(fetcher).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(1500);
    expect(fetcher).toHaveBeenCalledTimes(3);
    expect(handlers.done).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(5000);
    expect(fetcher).toHaveBeenCalledTimes(3);
    stop();
  });
  it("does not poll when idle", async () => {
    vi.useFakeTimers();
    const fetcher = vi.fn().mockResolvedValue(job({ status: "idle", done: 0, total: 0 }));
    vi.stubGlobal("fetch", fetcher);
    const handlers = { update: vi.fn(), done: vi.fn(), error: vi.fn() };
    const stop = watchNarration("job-1", handlers);
    await vi.advanceTimersByTimeAsync(0);
    await vi.advanceTimersByTimeAsync(10000);
    expect(fetcher).toHaveBeenCalledTimes(1);
    expect(handlers.update).toHaveBeenCalledWith({ status: "idle", done: 0, total: 0 });
    expect(handlers.done).not.toHaveBeenCalled();
    stop();
  });
  it("stops on error and passes the server message", async () => {
    vi.useFakeTimers();
    const fetcher = vi.fn().mockResolvedValue(job({ status: "error", done: 1, total: 3, message: "Ollama σταμάτησε" }));
    vi.stubGlobal("fetch", fetcher);
    const handlers = { update: vi.fn(), done: vi.fn(), error: vi.fn() };
    const stop = watchNarration("job-1", handlers);
    await vi.advanceTimersByTimeAsync(0);
    expect(handlers.error).toHaveBeenCalledWith("Ollama σταμάτησε");
    await vi.advanceTimersByTimeAsync(5000);
    expect(fetcher).toHaveBeenCalledTimes(1);
    stop();
  });
  it("stops polling when cleaned up", async () => {
    vi.useFakeTimers();
    const fetcher = vi.fn().mockResolvedValue(job({ status: "running", done: 1, total: 3 }));
    vi.stubGlobal("fetch", fetcher);
    const handlers = { update: vi.fn(), done: vi.fn(), error: vi.fn() };
    const stop = watchNarration("job-1", handlers);
    await vi.advanceTimersByTimeAsync(0);
    stop();
    await vi.advanceTimersByTimeAsync(5000);
    expect(fetcher).toHaveBeenCalledTimes(1);
  });
});
