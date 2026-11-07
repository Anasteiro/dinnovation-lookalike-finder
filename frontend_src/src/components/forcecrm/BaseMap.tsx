import { useEffect, useMemo, useRef, useState } from "react";
import { Minus, Plus, RotateCcw, X } from "lucide-react";
import type { GeoDensity, GeoOutline, Municipality, Place } from "@/lib/api";
import { COS38, IDENTITY, MAP_THEMES, layerOpacity, outlinePath, outlineProjection, panBy, scaleBar, zoomAt, zoomAtFactor, type MapTheme, type ViewTransform } from "@/lib/basemap";
import { number } from "./primitives";

const PIN_PATH = "M0 0 C-3 -4.2 -6.3 -7 -6.3 -11.7 A6.3 6.3 0 1 1 6.3 -11.7 C6.3 -7 3 -4.2 0 0 Z";
const CONTROL_STYLE = { background: "#ffffff", color: "#3a3a3a", boxShadow: "0 1px 4px rgba(0,0,0,0.3)" } as const;

function Cells({ data, projection, fill, lo, hi }: { data: GeoDensity; projection: ReturnType<typeof outlineProjection>; fill: string; lo: number; hi: number }) {
  const max = Math.max(0, ...data.cells.map(cell => cell.n));
  const cw = data.cell_deg * COS38 * projection.degPx;
  const ch = data.cell_deg * projection.degPx;
  return <>
    {data.cells.map((cell, i) => {
      const text = `${number(cell.n)} επιχειρήσεις · ${cell.lat.toLocaleString("el-GR")}, ${cell.lon.toLocaleString("el-GR")}`;
      return <rect key={`${cell.lat}-${cell.lon}-${i}`} x={projection.x(cell.lon) - cw / 2} y={projection.y(cell.lat) - ch / 2} width={cw * 0.94} height={ch * 0.94} rx={ch * 0.1} fill={fill} fillOpacity={layerOpacity(cell.n, max, lo, hi)}><title>{text}</title></rect>;
    })}
  </>;
}

export function BaseMap({ outline, market, overlay, overlayFill, municipalities, selected, pins, place, onPlace, theme, description }: {
  outline: GeoOutline;
  market: GeoDensity | null;
  overlay: GeoDensity | null;
  overlayFill: string;
  municipalities: Municipality[];
  selected: Municipality | null;
  pins: Place[] | null;
  place: Place | null;
  onPlace: (place: Place | null) => void;
  theme: MapTheme;
  description?: string;
}) {
  const colors = MAP_THEMES[theme];
  const projection = useMemo(() => outlineProjection(outline), [outline]);
  const land = useMemo(() => outlinePath(outline, projection), [outline, projection]);
  const [view, setView] = useState<ViewTransform>(IDENTITY);
  const containerRef = useRef<HTMLDivElement>(null);
  const pointers = useRef(new Map<number, { x: number; y: number; startX: number; startY: number }>());
  const captured = useRef(new Set<number>());
  const pinch = useRef(0);
  const labels = municipalities.slice(0, 8);
  const bar = scaleBar(projection, view.k);

  const toLocal = (clientX: number, clientY: number) => {
    const rect = containerRef.current!.getBoundingClientRect();
    const s = projection.width / rect.width;
    return { px: (clientX - rect.left) * s, py: (clientY - rect.top) * s, s };
  };

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const rect = el.getBoundingClientRect();
      const s = projection.width / rect.width;
      setView(v => zoomAt(v, (event.clientX - rect.left) * s, (event.clientY - rect.top) * s, event.deltaY, event.deltaMode));
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [projection]);

  const zoomButton = (factor: number) => setView(v => zoomAtFactor(v, projection.width / 2, projection.height / 2, factor));

  return <div ref={containerRef} className="relative select-none overflow-hidden rounded-md" style={{ touchAction: "none" }}
    onPointerDown={event => { if ((event.target as HTMLElement).closest("button")) return; pointers.current.set(event.pointerId, { x: event.clientX, y: event.clientY, startX: event.clientX, startY: event.clientY }); const pair = [...pointers.current.values()]; if (pair.length === 2 && pair[0] && pair[1]) pinch.current = Math.hypot(pair[0].x - pair[1].x, pair[0].y - pair[1].y); }}
    onPointerMove={event => {
      const previous = pointers.current.get(event.pointerId);
      if (!previous) return;
      if (!captured.current.has(event.pointerId)) {
        if (Math.hypot(event.clientX - previous.startX, event.clientY - previous.startY) < 4) return;
        event.currentTarget.setPointerCapture(event.pointerId);
        captured.current.add(event.pointerId);
      }
      pointers.current.set(event.pointerId, { ...previous, x: event.clientX, y: event.clientY });
      const rect = containerRef.current!.getBoundingClientRect();
      const s = projection.width / rect.width;
      if (pointers.current.size === 1) setView(v => panBy(v, (event.clientX - previous.x) * s, (event.clientY - previous.y) * s));
      else if (pointers.current.size === 2) {
        const pair = [...pointers.current.values()];
        const a = pair[0], b = pair[1];
        if (!a || !b) return;
        const dist = Math.hypot(a.x - b.x, a.y - b.y);
        if (pinch.current > 0) {
          const { px, py } = toLocal((a.x + b.x) / 2, (a.y + b.y) / 2);
          setView(v => zoomAtFactor(v, px, py, dist / pinch.current));
        }
        pinch.current = dist;
      }
    }}
    onPointerUp={event => { pointers.current.delete(event.pointerId); captured.current.delete(event.pointerId); pinch.current = 0; }}
    onPointerCancel={event => { pointers.current.delete(event.pointerId); captured.current.delete(event.pointerId); pinch.current = 0; }}
    onDoubleClick={event => { const { px, py } = toLocal(event.clientX, event.clientY); setView(v => zoomAtFactor(v, px, py, 1.6)); }}>
    <svg viewBox={`0 0 ${projection.width} ${projection.height}`} className="block w-full" style={{ aspectRatio: `${projection.width} / ${projection.height}` }} role="img" aria-label={description || "Χάρτης Ελλάδας με πυκνότητα επιχειρήσεων"}>
      <title>Βασικός χάρτης Ελλάδας με στρώματα επιχειρήσεων</title>
      <rect width={projection.width} height={projection.height} rx={10} fill={colors.sea} />
      <g transform={`translate(${view.x} ${view.y}) scale(${view.k})`}>
        <path d={land} fill={colors.land} fillRule="evenodd" stroke={colors.border} strokeWidth={0.6 / view.k} />
        {market && <Cells data={market} projection={projection} fill={colors.market} lo={0.15} hi={0.4} />}
        {overlay && <Cells data={overlay} projection={projection} fill={overlayFill} lo={0.5} hi={1} />}
        {selected && <circle data-testid="municipality-ring" cx={projection.x(selected.lon)} cy={projection.y(selected.lat)} r={14 / view.k} fill="none" stroke={colors.label} strokeWidth={2 / view.k}><title>{selected.municipality.split(" / ")[0]}</title></circle>}
        {labels.map((item, i) => <text key={`${item.municipality}-${i}`} x={projection.x(item.lon)} y={projection.y(item.lat)} textAnchor="middle" fontSize={11 / view.k} fill={colors.label} stroke={colors.halo} strokeWidth={3 / view.k} paintOrder="stroke" style={{ pointerEvents: "none" }}>{item.municipality.split(" / ")[0]}</text>)}
        {pins?.map((p, i) => <g key={`${p.name}-${i}`} transform={`translate(${projection.x(p.lon)} ${projection.y(p.lat)}) scale(${1 / view.k})`} className="cursor-pointer" onClick={event => { event.stopPropagation(); onPlace(p); }}>
          <path d={PIN_PATH} fill="#ea4335" stroke="#ffffff" strokeWidth={0.8} />
          <circle cx={0} cy={-11.7} r={2.2} fill="#ffffff" />
          <title>{`${p.name} · ★ ${p.rating ?? "—"} (${p.reviews ?? "—"})`}</title>
        </g>)}
      </g>
      <g transform={`translate(14 ${projection.height - 16})`} style={{ pointerEvents: "none" }}>
        <line x1={0} y1={0} x2={bar.px} y2={0} stroke={colors.label} strokeWidth={2} />
        <line x1={0} y1={-4} x2={0} y2={0} stroke={colors.label} strokeWidth={2} />
        <line x1={bar.px} y1={-4} x2={bar.px} y2={0} stroke={colors.label} strokeWidth={2} />
        <text x={bar.px / 2} y={-6} textAnchor="middle" fontSize={11} fill={colors.label} stroke={colors.halo} strokeWidth={3} paintOrder="stroke">{number(bar.km)} χλμ</text>
      </g>
    </svg>
    <div className="absolute right-2 top-2 flex flex-col gap-1.5">
      <button type="button" aria-label="Μεγέθυνση" onClick={() => zoomButton(1.5)} className="flex h-8 w-8 items-center justify-center rounded-md" style={CONTROL_STYLE}><Plus className="h-4 w-4" /></button>
      <button type="button" aria-label="Σμίκρυνση" onClick={() => zoomButton(1 / 1.5)} className="flex h-8 w-8 items-center justify-center rounded-md" style={CONTROL_STYLE}><Minus className="h-4 w-4" /></button>
      <button type="button" aria-label="Επαναφορά" onClick={() => setView(IDENTITY)} className="flex h-8 w-8 items-center justify-center rounded-md" style={CONTROL_STYLE}><RotateCcw className="h-4 w-4" /></button>
    </div>
    <p className="pointer-events-none absolute bottom-1.5 right-2 text-[10px]" style={{ color: colors.label, opacity: 0.75 }}>Όρια: Natural Earth (δημόσιο κτήμα)</p>
    {place && <div className="absolute bottom-8 left-2 max-w-xs rounded-md p-3 text-sm shadow-lg" style={{ background: colors.land, color: colors.label, border: `1px solid ${colors.border}` }} role="dialog" aria-label="Στοιχεία επιχείρησης">
      <div className="flex items-start justify-between gap-2">
        <p className="font-semibold">{place.name}</p>
        <button type="button" aria-label="Κλείσιμο" onClick={() => onPlace(null)} className="shrink-0 opacity-70"><X className="h-4 w-4" /></button>
      </div>
      <p className="mt-1 text-xs">★ {place.rating ?? "—"} ({number(place.reviews ?? 0)} κριτικές)</p>
    </div>}
  </div>;
}
