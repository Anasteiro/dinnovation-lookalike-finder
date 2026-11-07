import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { ChatPanel } from "../components/forcecrm/ChatPanel";

const answer = { question: "q", text: "Η ανάλυση βρήκε 2 προφίλ.", kind: "answer", rows: [{ "Θέση": 1, "Επωνυμία": "PARCO", company_id: "42" }], columns: ["Θέση", "Επωνυμία"], sources: ["profiles.json"], filters: "Νομός ΧΑΝΙΩΝ", ai_generated: false, ai_summary: null, llm_available: false, seconds: 0.004 };

class RO { observe() {} unobserve() {} disconnect() {} }
vi.stubGlobal("ResizeObserver", RO);

describe("Chat panel", () => {
  afterEach(() => { vi.restoreAllMocks(); });
  it("sends a question and renders a grounded answer with its source", async () => {
    const f = vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify(answer), { status: 200 }));
    render(<ChatPanel job="j1" />);
    fireEvent.change(screen.getByLabelText("Ερώτηση"), { target: { value: "Ποια προφίλ βρήκες;" } });
    fireEvent.click(screen.getByLabelText("Αποστολή"));
    await waitFor(() => expect(screen.getByText("Η ανάλυση βρήκε 2 προφίλ.")).toBeTruthy());
    expect(screen.getByText(/Πηγή: profiles.json/)).toBeTruthy();
    expect(screen.getByText("Νομός ΧΑΝΙΩΝ")).toBeTruthy();
    const body = JSON.parse(String((f.mock.calls[0]?.[1] as RequestInit).body));
    expect(body).toEqual({ question: "Ποια προφίλ βρήκες;", ai_summary: false });
  });
  it("caps the question length", () => {
    render(<ChatPanel job="j1" />);
    expect((screen.getByLabelText("Ερώτηση") as HTMLTextAreaElement).maxLength).toBe(400);
  });
});
