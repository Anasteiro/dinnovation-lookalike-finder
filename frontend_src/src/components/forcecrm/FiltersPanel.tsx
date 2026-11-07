import { useEffect, useMemo, useState } from "react";
import { Download, Filter, RotateCcw, SlidersHorizontal, ArrowUp, ArrowDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { api, type FactorsResponse, type Profile, type RefineFilters, type RefineResponse } from "@/lib/api";
import { cn } from "@/lib/utils";
import { ErrorBanner, Panel } from "./primitives";
import { ClarityBadge } from "./Clarity";

const fmt = (v: unknown, digits = 1) => typeof v === "number" && Number.isFinite(v) ? v.toLocaleString("el-GR", { maximumFractionDigits: digits }) : "—";
const LIMIT = 25;

export function FiltersPanel({ job, profiles, profile, onProfileChange }: { job: string; profiles: Profile[]; profile: string; onProfileChange: (id: string) => void }) {
  const [meta, setMeta] = useState<FactorsResponse | null>(null);
  const [legal, setLegal] = useState<string[]>([]);
  const [prefs, setPrefs] = useState<string[]>([]);
  const [prefQuery, setPrefQuery] = useState("");
  const [minClarity, setMinClarity] = useState(0);
  const [foundedFrom, setFoundedFrom] = useState<string>("");
  const [foundedTo, setFoundedTo] = useState<string>("");
  const [noBranches, setNoBranches] = useState(false);
  const [ranges, setRanges] = useState<Record<string, { min: string; max: string }>>({});
  const [boosts, setBoosts] = useState<Record<string, number>>({});
  const [strength, setStrength] = useState(0.5);
  const [offset, setOffset] = useState(0);
  const [res, setRes] = useState<RefineResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => { const c = new AbortController(); api.factors(job, c.signal).then(m => { if (!c.signal.aborted) setMeta(m); }).catch((e: Error) => { if (!c.signal.aborted) setError(e.message); }); return () => c.abort(); }, [job]);

  const filters: RefineFilters = useMemo(() => {
    const f: RefineFilters = {};
    if (legal.length) f.legal = legal;
    if (prefs.length) f.prefectures = prefs;
    if (minClarity > 0) f.min_clarity = minClarity;
    if (foundedFrom) f.founded_from = Number(foundedFrom);
    if (foundedTo) f.founded_to = Number(foundedTo);
    if (noBranches) f.exclude_branches = true;
    const fr: Record<string, { min?: number | null; max?: number | null }> = {};
    for (const [k, r] of Object.entries(ranges)) { const mn = r.min.trim() === "" ? null : Number(r.min.replace(",", ".")); const mx = r.max.trim() === "" ? null : Number(r.max.replace(",", ".")); if (mn !== null || mx !== null) fr[k] = { min: mn, max: mx }; }
    if (Object.keys(fr).length) f.factors = fr;
    return f;
  }, [legal, prefs, minClarity, foundedFrom, foundedTo, noBranches, ranges]);

  useEffect(() => { setOffset(0); }, [filters, boosts, strength, profile]);
  useEffect(() => {
    const c = new AbortController();
    const t = window.setTimeout(() => {
      setLoading(true); setError(null);
      api.refine(job, { profile, filters, boosts, strength, limit: LIMIT, offset }, c.signal)
        .then(r => { if (!c.signal.aborted) setRes(r); }).catch((e: Error) => { if (!c.signal.aborted && e.name !== "AbortError") setError(e.message); })
        .finally(() => { if (!c.signal.aborted) setLoading(false); });
    }, 350);
    return () => { window.clearTimeout(t); c.abort(); };
  }, [job, profile, filters, boosts, strength, offset]);

  const reset = () => { setLegal([]); setPrefs([]); setMinClarity(0); setFoundedFrom(""); setFoundedTo(""); setNoBranches(false); setRanges({}); setBoosts({}); setStrength(0.5); };
  const factors = meta?.factors ?? [];
  const activeBoosts = Object.values(boosts).some(v => Math.abs(v) > 0);
  const profFactors = meta?.profiles.find(p => p.profile_id === profile)?.external_factors ?? [];
  const removedEntries = Object.entries(res?.removed ?? {}).filter(([, n]) => n > 0);
  const toggle = (arr: string[], v: string, set: (x: string[]) => void) => set(arr.includes(v) ? arr.filter(x => x !== v) : [...arr, v]);
  const prefList = (meta?.options.prefectures ?? []).filter(p => p.toLowerCase().includes(prefQuery.toLowerCase()));

  return <div className="grid gap-6 xl:grid-cols-[380px_minmax(0,1fr)]">
    <div className="space-y-4">
      <Panel className="p-5"><div className="flex items-center justify-between gap-2"><h2 className="flex items-center gap-2 font-semibold"><Filter className="h-4 w-4" />Φίλτρα</h2><Button variant="outline" size="sm" onClick={reset}><RotateCcw className="h-4 w-4" />Επαναφορά</Button></div>
        <label className="mt-4 block text-xs text-muted-foreground">Προφίλ</label>
        <Select value={profile} onValueChange={onProfileChange}><SelectTrigger aria-label="Προφίλ" className="mt-1"><SelectValue /></SelectTrigger><SelectContent><SelectItem value="all">Όλα τα προφίλ (συνολική λίστα)</SelectItem>{profiles.map(p => <SelectItem key={p.profile_id} value={p.profile_id}>{p.title}</SelectItem>)}</SelectContent></Select>
        <p className="mt-4 text-xs text-muted-foreground">Νομική μορφή</p>
        <div className="mt-2 grid grid-cols-2 gap-2">{(meta?.options.legal ?? []).slice(0, 10).map(o => <label key={o.value} className="flex items-center gap-2 text-sm"><Checkbox checked={legal.includes(o.value)} onCheckedChange={() => toggle(legal, o.value, setLegal)} aria-label={o.value} /><span className="truncate">{o.value}</span><span className="ml-auto font-mono text-[10px] text-muted-foreground">{o.count.toLocaleString("el-GR")}</span></label>)}</div>
        <p className="mt-4 text-xs text-muted-foreground">Νομοί {prefs.length > 0 && <span className="text-primary">({prefs.length} επιλεγμένοι)</span>}</p>
        <Input className="mt-2" placeholder="Αναζήτηση νομού…" value={prefQuery} onChange={e => setPrefQuery(e.target.value)} aria-label="Αναζήτηση νομού" />
        <div className="mt-2 max-h-36 space-y-1 overflow-y-auto pr-1">{prefList.map(p => <label key={p} className="flex items-center gap-2 text-sm"><Checkbox checked={prefs.includes(p)} onCheckedChange={() => toggle(prefs, p, setPrefs)} aria-label={p} /><span className="truncate">{p}</span></label>)}</div>
        <div className="mt-4 flex items-center justify-between text-xs text-muted-foreground"><span>Ελάχιστη καθαρότητα ΚΑΔ</span><span className="font-mono text-foreground">{minClarity}</span></div>
        <Slider className="mt-2" min={0} max={100} step={5} value={[minClarity]} onValueChange={v => setMinClarity(v[0] ?? 0)} aria-label="Ελάχιστη καθαρότητα ΚΑΔ" />
        <div className="mt-4 grid grid-cols-2 gap-2"><div><label className="text-xs text-muted-foreground">Ίδρυση από</label><Input inputMode="numeric" placeholder={meta?.options.founded ? String(meta.options.founded[0]) : ""} value={foundedFrom} onChange={e => setFoundedFrom(e.target.value.replace(/\D/g, ""))} /></div><div><label className="text-xs text-muted-foreground">έως</label><Input inputMode="numeric" placeholder={meta?.options.founded ? String(meta.options.founded[1]) : ""} value={foundedTo} onChange={e => setFoundedTo(e.target.value.replace(/\D/g, ""))} /></div></div>
        <label className="mt-4 flex items-center justify-between gap-2 text-sm"><span>Χωρίς υποκαταστήματα</span><Switch checked={noBranches} onCheckedChange={setNoBranches} aria-label="Χωρίς υποκαταστήματα" /></label>
      </Panel>
      <Panel className="p-5"><h2 className="flex items-center gap-2 font-semibold"><SlidersHorizontal className="h-4 w-4" />Πρόσθετοι παράγοντες</h2>
        <p className="mt-1 text-xs text-muted-foreground">Από ανοικτά δημόσια δεδομένα ανά δήμο. Ορίστε όρια (φίλτρο) ή/και προτίμηση (αλλάζει τη σειρά).</p>
        <div className="mt-4 space-y-5">{factors.map(f => { const r = ranges[f.key] ?? { min: "", max: "" }; const b = boosts[f.key] ?? 0; return <div key={f.key} className="border-t border-border pt-4 first:border-0 first:pt-0">
          <p className="text-sm font-medium">{f.label}</p><p className="text-[11px] text-muted-foreground">Αγορά: διάμεσος {fmt(f.market_median, 2)} {f.unit} · εύρος {fmt(f.min, 2)}–{fmt(f.max, 2)}</p>
          <div className="mt-2 grid grid-cols-2 gap-2"><Input aria-label={`${f.label} ελάχιστο`} placeholder="ελάχιστο" value={r.min} onChange={e => setRanges({ ...ranges, [f.key]: { ...r, min: e.target.value } })} /><Input aria-label={`${f.label} μέγιστο`} placeholder="μέγιστο" value={r.max} onChange={e => setRanges({ ...ranges, [f.key]: { ...r, max: e.target.value } })} /></div>
          <div className="mt-3 flex items-center justify-between text-[11px] text-muted-foreground"><span className="flex items-center gap-1"><ArrowDown className="h-3 w-3" />χαμηλότερο</span><span className={cn("font-mono", b !== 0 && "text-primary")}>{b === 0 ? "ουδέτερο" : b > 0 ? `προτίμηση +${b.toFixed(1)}` : `προτίμηση ${b.toFixed(1)}`}</span><span className="flex items-center gap-1">υψηλότερο<ArrowUp className="h-3 w-3" /></span></div>
          <Slider className="mt-1" min={-1} max={1} step={0.1} value={[b]} onValueChange={v => setBoosts({ ...boosts, [f.key]: Math.round((v[0] ?? 0) * 10) / 10 })} aria-label={`Προτίμηση για ${f.label}`} />
          <p className="mt-1 text-[10px] text-muted-foreground">Πηγή: {f.source} · κάλυψη {Math.round(f.coverage * 100)}%</p>
        </div>; })}</div>
        <div className="mt-5 border-t border-border pt-4"><div className="flex items-center justify-between text-xs text-muted-foreground"><span>Βάρος προτιμήσεων έναντι μοντέλου</span><span className="font-mono text-foreground">{Math.round(strength * 100)}%</span></div><Slider className="mt-2" min={0} max={1} step={0.05} value={[strength]} onValueChange={v => setStrength(v[0] ?? 0.5)} aria-label="Βάρος προτιμήσεων" disabled={!activeBoosts} /><p className="mt-1 text-[10px] text-muted-foreground">Χωρίς προτιμήσεις η σειρά είναι αυτή του μοντέλου.</p></div>
      </Panel>
    </div>
    <div className="min-w-0 space-y-4">
      {profile !== "all" && profFactors.length > 0 && <Panel className="p-5"><h2 className="font-semibold">Οι πελάτες του προφίλ σε σχέση με την αγορά</h2><div className="mt-3 grid gap-3 md:grid-cols-3">{profFactors.map(pf => <div key={pf.key} className="rounded-lg border border-border p-3"><p className="text-xs text-muted-foreground">{pf.label}</p><p className="mt-1 font-mono text-lg">{fmt(pf.profile_median, 2)}</p><p className="text-[11px] text-muted-foreground">αγορά {fmt(pf.market_median, 2)} {pf.unit}</p><p className={cn("mt-1 text-[11px]", pf.significant ? "text-success" : "text-muted-foreground")}>{pf.significant ? "στατιστικά σημαντική διαφορά" : "χωρίς σημαντική διαφορά"} (p={pf.p_value.toExponential(1)})</p></div>)}</div></Panel>}
      <Panel className="p-5"><div className="flex flex-wrap items-center justify-between gap-3"><div><h2 className="font-semibold">Αποτελέσματα</h2><p className="text-sm text-muted-foreground">{res ? `${res.total.toLocaleString("el-GR")} επιχειρήσεις μένουν μετά τα φίλτρα` : "—"}{loading && " · ενημέρωση…"}</p></div><Button variant="outline" onClick={() => api.download(job, "refined_lookalikes.csv").catch((e: Error) => setError(e.message))} disabled={!res}><Download className="h-4 w-4" />Εξαγωγή λίστας CSV</Button></div>
        {removedEntries.length > 0 && <div className="mt-3 flex flex-wrap gap-2">{removedEntries.map(([k, n]) => <span key={k} className="rounded-full border border-border px-2 py-0.5 text-xs text-muted-foreground">−{n.toLocaleString("el-GR")} από «{k}»</span>)}</div>}
        {res && <div className="mt-3 grid gap-2 sm:grid-cols-3">{factors.map(f => <div key={f.key} className="rounded-lg bg-muted/40 p-2 text-xs"><span className="text-muted-foreground">{f.label} (διάμεσος top 2000): </span><span className="font-mono">{fmt(res.top2000_medians[f.key], 2)}</span></div>)}</div>}
        {error && <div className="mt-3"><ErrorBanner message={error} /></div>}
        <div className="mt-4 overflow-x-auto border-y border-border"><Table><TableHeader><TableRow><TableHead>#</TableHead><TableHead>Θέση μοντέλου</TableHead><TableHead>Επωνυμία</TableHead><TableHead>Δήμος</TableHead><TableHead>Μορφή</TableHead><TableHead>Καθαρότητα ΚΑΔ</TableHead>{factors.map(f => <TableHead key={f.key} className="whitespace-normal text-xs">{f.label}</TableHead>)}</TableRow></TableHeader>
          <TableBody>{res && res.items.length === 0 && <TableRow><TableCell colSpan={6 + factors.length} className="py-10 text-center text-muted-foreground">Καμία επιχείρηση δεν ικανοποιεί τα φίλτρα.</TableCell></TableRow>}{res?.items.map(r => { const moved = r.model_rank - r.rank; return <TableRow key={String(r.company_id)}><TableCell className="font-mono">{r.rank}</TableCell><TableCell className="font-mono text-xs text-muted-foreground">{r.model_rank}{activeBoosts && moved !== 0 && <span className={cn("ml-1", moved > 0 ? "text-success" : "text-destructive")}>{moved > 0 ? `▲${moved}` : `▼${-moved}`}</span>}</TableCell><TableCell className="min-w-40 whitespace-normal font-medium">{r.name}</TableCell><TableCell className="whitespace-normal text-xs text-muted-foreground">{r.municipality ?? "—"}</TableCell><TableCell className="text-xs">{r.legal_type ?? "—"}</TableCell><TableCell><ClarityBadge value={r.clarity} level={r.clarity_level} /></TableCell>{factors.map(f => <TableCell key={f.key} className="font-mono text-xs">{fmt(r[f.key], f.key === "muni_population" ? 0 : 2)}</TableCell>)}</TableRow>; })}</TableBody></Table></div>
        <div className="mt-3 flex items-center justify-between text-xs text-muted-foreground"><span>{res ? `${res.items.length ? offset + 1 : 0}–${offset + res.items.length} από ${res.total.toLocaleString("el-GR")}` : ""}</span><div className="flex gap-2"><Button variant="outline" size="sm" disabled={loading || offset === 0} onClick={() => setOffset(Math.max(0, offset - LIMIT))}>Προηγούμενα</Button><Button variant="outline" size="sm" disabled={loading || !res || offset + LIMIT >= res.total} onClick={() => setOffset(offset + LIMIT)}>Επόμενα</Button></div></div>
        <p className="mt-3 text-[11px] text-muted-foreground">{meta?.note}</p>
      </Panel>
    </div>
  </div>;
}
