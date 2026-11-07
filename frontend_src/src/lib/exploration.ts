import { api, type Multipliers, type WhatIfResult, type GeoDensity } from "./api";

export const DEFAULT_MULTIPLIERS: Multipliers = { activity: 1, geo: 1, legal: 1, age: 1 };
export const WEIGHT_RANGE = { min: 0, max: 2, step: 0.1 };

export function scheduleWhatIf(job: string, weights: Multipliers, evaluate: boolean, handlers: { result: (result: WhatIfResult) => void; error: (message: string) => void; started: () => void }, delay = 600) {
  const controller = new AbortController();
  const timer = setTimeout(async () => {
    handlers.started();
    try {
      const result = await api.whatif(job, weights, evaluate, controller.signal);
      if (!controller.signal.aborted) handlers.result(result);
    } catch (error) {
      if (!controller.signal.aborted) handlers.error(error instanceof Error ? error.message : String(error));
    }
  }, delay);
  return () => { clearTimeout(timer); controller.abort(); };
}

export function geoProjection(data: GeoDensity) {
  const cos = Math.cos(38 * Math.PI / 180);
  const lonMin = data.bounds.lon[0] - data.cell_deg / 2;
  const latMax = data.bounds.lat[1] + data.cell_deg / 2;
  const width = Math.max(data.cell_deg, data.bounds.lon[1] - data.bounds.lon[0] + data.cell_deg) * cos;
  const height = Math.max(data.cell_deg, data.bounds.lat[1] - data.bounds.lat[0] + data.cell_deg);
  return { width, height, cellWidth: data.cell_deg * cos, cellHeight: data.cell_deg, x: (lon: number) => (lon - lonMin) * cos, y: (lat: number) => latMax - lat };
}

export const densityOpacity = (n: number, max: number) => 0.15 + 0.8 * Math.sqrt(max > 0 ? Math.max(0, Math.min(1, n / max)) : 0);
export const PROFILE_FILLS = ["fill-chart-1", "fill-chart-2", "fill-chart-3", "fill-chart-4", "fill-chart-5"];