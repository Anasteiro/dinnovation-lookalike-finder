import { describe, expect, it } from "vitest";
import { IDENTITY, MAX_ZOOM, MIN_ZOOM, layerOpacity, outlinePath, outlineProjection, panBy, scaleBar, zoomAt, zoomAtFactor } from "./basemap";
import type { GeoOutline } from "./api";

const outline: GeoOutline = {
  type: "Feature",
  geometry: { type: "MultiPolygon", coordinates: [[[[20, 35], [28, 35], [28, 41], [20, 41], [20, 35]]], [[[23, 36], [24, 36], [24, 37], [23, 37], [23, 36]]]] },
};

describe("outlineProjection", () => {
  it("keeps the aspect ratio and fits the target size", () => {
    const p = outlineProjection(outline, 900);
    const spanX = 8 * Math.cos(38 * Math.PI / 180);
    const spanY = 6;
    expect(p.width / p.height).toBeCloseTo(spanX / spanY, 6);
    expect(Math.max(p.width, p.height)).toBeCloseTo(900, 6);
  });
  it("maps bounds corners to the viewBox corners", () => {
    const p = outlineProjection(outline, 900);
    expect(p.x(20)).toBeCloseTo(0, 6);
    expect(p.y(41)).toBeCloseTo(0, 6);
    expect(p.x(28)).toBeCloseTo(p.width, 6);
    expect(p.y(35)).toBeCloseTo(p.height, 6);
  });
});

describe("outlinePath", () => {
  it("emits one subpath per ring", () => {
    const d = outlinePath(outline, outlineProjection(outline));
    expect(d.startsWith("M")).toBe(true);
    expect(d.match(/M/g)).toHaveLength(2);
    expect(d.endsWith("Z")).toBe(true);
  });
});

describe("zoomAt", () => {
  it("keeps the world point under the cursor stationary", () => {
    const view = { k: 2, x: 30, y: -10 };
    const px = 120, py = 80;
    const worldX = (px - view.x) / view.k, worldY = (py - view.y) / view.k;
    const next = zoomAt(view, px, py, -120);
    expect((px - next.x) / next.k).toBeCloseTo(worldX, 6);
    expect((py - next.y) / next.k).toBeCloseTo(worldY, 6);
    expect(next.k).toBeGreaterThan(view.k);
  });
  it("clamps zoom between 1 and 12", () => {
    let view = IDENTITY;
    for (let i = 0; i < 200; i++) view = zoomAt(view, 100, 100, -500);
    expect(view.k).toBe(MAX_ZOOM);
    for (let i = 0; i < 400; i++) view = zoomAt(view, 100, 100, 500);
    expect(view.k).toBe(MIN_ZOOM);
  });
  it("normalizes line deltaMode like 16px per line", () => {
    const byLines = zoomAt(IDENTITY, 0, 0, -3, 1);
    const byPixels = zoomAt(IDENTITY, 0, 0, -48, 0);
    expect(byLines.k).toBeCloseTo(byPixels.k, 6);
  });
});

describe("zoomAtFactor and panBy", () => {
  it("applies a direct factor anchored at a point", () => {
    const next = zoomAtFactor(IDENTITY, 50, 50, 2);
    expect(next.k).toBe(2);
    expect(next.x).toBeCloseTo(-50, 6);
    expect(next.y).toBeCloseTo(-50, 6);
  });
  it("pans by deltas without changing zoom", () => {
    const next = panBy({ k: 3, x: 1, y: 2 }, 10, -5);
    expect(next).toEqual({ k: 3, x: 11, y: -3 });
  });
});

describe("scaleBar", () => {
  it("picks a nice kilometre length near the target width", () => {
    const p = outlineProjection(outline, 900);
    const bar = scaleBar(p, 1, 110);
    expect([1, 2, 5, 10, 20, 50, 100, 200, 500]).toContain(bar.km);
    expect(bar.px).toBeGreaterThan(0);
    expect(bar.px).toBeLessThan(300);
  });
  it("shrinks in kilometres as zoom grows", () => {
    const p = outlineProjection(outline, 900);
    expect(scaleBar(p, 12).km).toBeLessThanOrEqual(scaleBar(p, 1).km);
  });
});

describe("layerOpacity", () => {
  it("stays inside the layer range", () => {
    expect(layerOpacity(0, 100, 0.15, 0.4)).toBeCloseTo(0.15, 6);
    expect(layerOpacity(100, 100, 0.15, 0.4)).toBeCloseTo(0.4, 6);
    expect(layerOpacity(50, 100, 0.5, 1)).toBeGreaterThan(0.5);
    expect(layerOpacity(50, 100, 0.5, 1)).toBeLessThan(1);
  });
});
