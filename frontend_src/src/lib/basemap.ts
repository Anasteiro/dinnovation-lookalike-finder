import type { GeoOutline } from "./api";

export const COS38 = Math.cos(38 * Math.PI / 180);
export const MIN_ZOOM = 1;
export const MAX_ZOOM = 12;

export const MAP_THEMES = {
  light: { sea: "#aad3df", land: "#f2efe9", border: "#c9c2b2", halo: "#ffffff", label: "#3f3a32", market: "#8b949e" },
  dark: { sea: "#1b2b3a", land: "#2a2f3a", border: "#3b4252", halo: "#1b2b3a", label: "#d7dce4", market: "#5f6b7a" },
} as const;
export type MapTheme = keyof typeof MAP_THEMES;

export type OutlineProjection = {
  width: number;
  height: number;
  degPx: number;
  x: (lon: number) => number;
  y: (lat: number) => number;
};

export function outlineProjection(outline: GeoOutline, size = 900): OutlineProjection {
  let lonMin = Infinity, lonMax = -Infinity, latMin = Infinity, latMax = -Infinity;
  for (const polygon of outline.geometry.coordinates)
    for (const ring of polygon)
      for (const point of ring) {
        const lon = point[0] ?? 0, lat = point[1] ?? 0;
        if (lon < lonMin) lonMin = lon;
        if (lon > lonMax) lonMax = lon;
        if (lat < latMin) latMin = lat;
        if (lat > latMax) latMax = lat;
      }
  if (!Number.isFinite(lonMin)) { lonMin = 19; lonMax = 30; latMin = 34; latMax = 42; }
  const spanX = Math.max(0.01, (lonMax - lonMin) * COS38);
  const spanY = Math.max(0.01, latMax - latMin);
  const degPx = size / Math.max(spanX, spanY);
  return {
    width: spanX * degPx,
    height: spanY * degPx,
    degPx,
    x: (lon: number) => (lon - lonMin) * COS38 * degPx,
    y: (lat: number) => (latMax - lat) * degPx,
  };
}

export function outlinePath(outline: GeoOutline, projection: OutlineProjection): string {
  const parts: string[] = [];
  for (const polygon of outline.geometry.coordinates)
    for (const ring of polygon)
      ring.forEach((point, i) => {
        parts.push(`${i === 0 ? "M" : "L"}${projection.x(point[0] ?? 0).toFixed(2)} ${projection.y(point[1] ?? 0).toFixed(2)}`);
      });
  return parts.join(" ") + " Z";
}

export type ViewTransform = { k: number; x: number; y: number };
export const IDENTITY: ViewTransform = { k: 1, x: 0, y: 0 };

export function zoomAtFactor(view: ViewTransform, px: number, py: number, factor: number): ViewTransform {
  const k = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, view.k * factor));
  const r = k / view.k;
  return { k, x: px - (px - view.x) * r, y: py - (py - view.y) * r };
}

export function zoomAt(view: ViewTransform, px: number, py: number, deltaY: number, deltaMode = 0): ViewTransform {
  const dy = deltaY * (deltaMode === 1 ? 16 : deltaMode === 2 ? 100 : 1);
  return zoomAtFactor(view, px, py, Math.exp(-dy * 0.0015));
}

export function panBy(view: ViewTransform, dx: number, dy: number): ViewTransform {
  return { ...view, x: view.x + dx, y: view.y + dy };
}

const SCALE_STEPS = [1, 2, 5, 10, 20, 50, 100, 200, 500];

export function scaleBar(projection: OutlineProjection, k: number, targetPx = 110): { km: number; px: number } {
  const kmPerPx = 111 / (projection.degPx * k);
  let best: number = SCALE_STEPS[0] ?? 1;
  for (const step of SCALE_STEPS)
    if (Math.abs(step / kmPerPx - targetPx) < Math.abs(best / kmPerPx - targetPx)) best = step;
  return { km: best, px: best / kmPerPx };
}

export function layerOpacity(n: number, max: number, lo: number, hi: number): number {
  const t = max > 0 ? Math.min(1, Math.max(0, n / max)) : 0;
  return lo + (hi - lo) * Math.sqrt(t);
}
