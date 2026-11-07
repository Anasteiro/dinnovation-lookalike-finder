import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ClarityBadge, clarityTone } from "../components/forcecrm/Clarity";

describe("Activity clarity badge", () => {
  it("shows value and level with an accessible label", () => {
    render(<ClarityBadge value={23} level="Θορυβώδες" />);
    expect(screen.getByLabelText("Καθαρότητα ΚΑΔ 23 από 100, Θορυβώδες")).toBeTruthy();
  });
  it("shows a dash when the index is missing", () => {
    render(<ClarityBadge value={null} level={null} />);
    expect(screen.getByText("—")).toBeTruthy();
  });
  it("uses distinct tones per level", () => {
    expect(new Set(["Καθαρό", "Μέτριο", "Θορυβώδες"].map(clarityTone)).size).toBe(3);
  });
});
