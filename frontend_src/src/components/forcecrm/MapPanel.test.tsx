import { describe, expect, it, vi, beforeAll } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { MapPanel } from "./MapPanel";
import type { Profile } from "@/lib/api";

beforeAll(() => { globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} } as unknown as typeof ResizeObserver; });

describe("MapPanel (simplified)", () => {
  it("shows a plain summary, legend, areas list and keeps advanced settings folded", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const u = String(input);
      const body = u.includes("/api/geo/outline") ? { type: "Feature", geometry: { type: "MultiPolygon", coordinates: [[[[20, 35], [28, 35], [28, 41], [20, 41], [20, 35]]]] } }
        : u.includes("/api/geo/density") ? { cell_deg: 0.1, total: 120, bounds: { lat: [35, 41], lon: [20, 28] }, cells: [{ lat: 37.9, lon: 23.7, n: 80 }, { lat: 40.6, lon: 22.9, n: 40 }] }
        : u.includes("/api/geo/municipalities") ? { items: [{ municipality: "ΑΘΗΝΑΙΩΝ / ΑΤΤΙΚΗΣ", lookalikes: 80, market: 900, share_of_market_in_top: 0.1, lat: 37.9, lon: 23.7 }, { municipality: "ΘΕΣΣΑΛΟΝΙΚΗΣ", lookalikes: 40, market: 500, share_of_market_in_top: 0.1, lat: 40.6, lon: 22.9 }] }
        : { available: false };
      return new Response(JSON.stringify(body), { status: 200 });
    });
    render(<MapPanel job="j" profiles={[{ profile_id: "icp-1", title: "Ταβέρνες" }] as unknown as Profile[]} profile="all" onProfileChange={() => {}} />);
    expect(await screen.findByText(/Οι περισσότερες προτάσεις βρίσκονται: ΑΘΗΝΑΙΩΝ \(80\), ΘΕΣΣΑΛΟΝΙΚΗΣ \(40\)/)).toBeTruthy();
    expect(screen.getByLabelText("Υπόμνημα χάρτη").textContent).toMatch(/πιο έντονο χρώμα = περισσότερες/);
    expect(screen.getByRole("img").getAttribute("aria-label")).toMatch(/ΑΘΗΝΑΙΩΝ/);
    expect(screen.getByText("Περισσότερες ρυθμίσεις")).toBeTruthy();
    const area = screen.getByRole("button", { name: /ΑΘΗΝΑΙΩΝ: 80 προτάσεις/ });
    fireEvent.click(area); expect(area.getAttribute("aria-pressed")).toBe("true");
    expect(screen.getByText("Όλη η αγορά")).toBeTruthy();
  });
});
