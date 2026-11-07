import { useEffect, useState } from "react";
import { RotateCcw, MapPin } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { api, type GeoDensity, type GeoKind, type GeoOutline, type Municipality, type Place, type Profile } from "@/lib/api";
import type { MapTheme } from "@/lib/basemap";
import { cn } from "@/lib/utils";
import { BaseMap } from "./BaseMap";
import { ErrorBanner, Panel, number } from "./primitives";

const LAYERS = [
  { value: "lookalikes", label: "Προτάσεις", hint: "Πού βρίσκονται οι επιχειρήσεις που μοιάζουν με τους πελάτες σας." },
  { value: "clients", label: "Πελάτες", hint: "Πού βρίσκονται οι σημερινοί πελάτες σας." },
  { value: "market", label: "Όλη η αγορά", hint: "Όλες οι επιχειρήσεις εστίασης της χώρας." },
] as const;
const RESOLUTIONS = [{ value: "0.2", label: "Αδρή (~22 km)" }, { value: "0.1", label: "Μεσαία (~11 km)" }, { value: "0.05", label: "Λεπτομερής (~5 km)" }];
const THEMES = [{ value: "light", label: "Ανοιχτός" }, { value: "dark", label: "Σκούρος" }] as const;
const short = (m: string) => m.split(" / ")[0];

/** Map page, simplified: one choice (what to show) + profile, a plain-language summary and legend, and the areas list as
 *  the accessible alternative. Resolution, theme and Google pins are under "Περισσότερες ρυθμίσεις". */
export function MapPanel({ job, profiles, profile, onProfileChange }: { job: string; profiles: Profile[]; profile: string; onProfileChange: (value: string) => void }) {
  const [kind, setKind] = useState<GeoKind>("lookalikes");
  const [cell, setCell] = useState("0.1");
  const [theme, setTheme] = useState<MapTheme>("light");
  const [outline, setOutline] = useState<GeoOutline | null>(null);
  const [outlineError, setOutlineError] = useState<string | null>(null);
  const [density, setDensity] = useState<GeoDensity | null>(null);
  const [market, setMarket] = useState<GeoDensity | null>(null);
  const [municipalities, setMunicipalities] = useState<Municipality[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [municipalityError, setMunicipalityError] = useState<string | null>(null);
  const [retry, setRetry] = useState(0);
  const [selected, setSelected] = useState<Municipality | null>(null);
  const [pinsAvailable, setPinsAvailable] = useState(false);
  const [pinsOn, setPinsOn] = useState(false);
  const [places, setPlaces] = useState<Place[] | null>(null);
  const [place, setPlace] = useState<Place | null>(null);

  useEffect(() => {
    const controller = new AbortController(); setOutlineError(null);
    api.outline(controller.signal).then(value => { if (!controller.signal.aborted) setOutline(value); }).catch(e => { if (!controller.signal.aborted) setOutlineError(e instanceof Error ? e.message : String(e)); });
    return () => controller.abort();
  }, [retry]);
  useEffect(() => {
    const controller = new AbortController(); setDensity(null); setError(null);
    api.density(job, kind, profile, Number(cell), controller.signal).then(value => { if (!controller.signal.aborted) setDensity(value); }).catch(e => { if (!controller.signal.aborted) setError(e instanceof Error ? e.message : String(e)); });
    return () => controller.abort();
  }, [job, kind, profile, cell, retry]);
  useEffect(() => {
    if (kind === "market") { setMarket(null); return; }
    const controller = new AbortController(); setMarket(null);
    api.density(job, "market", profile, Number(cell), controller.signal).then(value => { if (!controller.signal.aborted) setMarket(value); }).catch(() => { if (!controller.signal.aborted) setMarket(null); });
    return () => controller.abort();
  }, [job, kind, profile, cell, retry]);
  useEffect(() => {
    const controller = new AbortController(); setMunicipalities(null); setMunicipalityError(null); setSelected(null);
    api.municipalities(job, profile, controller.signal).then(value => { if (!controller.signal.aborted) setMunicipalities(value.items); }).catch(e => { if (!controller.signal.aborted) setMunicipalityError(e instanceof Error ? e.message : String(e)); });
    return () => controller.abort();
  }, [job, profile, retry]);
  useEffect(() => {
    const controller = new AbortController();
    api.enrichmentStatus(controller.signal).then(value => { if (!controller.signal.aborted) setPinsAvailable(Boolean(value.available)); }).catch(() => { if (!controller.signal.aborted) setPinsAvailable(false); });
    return () => controller.abort();
  }, [job]);
  useEffect(() => {
    if (!pinsOn || !pinsAvailable) { setPlaces(null); setPlace(null); return; }
    const controller = new AbortController(); setPlaces(null); setPlace(null);
    api.places(job, 300, controller.signal).then(value => { if (!controller.signal.aborted) setPlaces(value); }).catch(() => { if (!controller.signal.aborted) setPlaces(null); });
    return () => controller.abort();
  }, [job, pinsOn, pinsAvailable]);

  const profileIndex = Math.max(0, profiles.findIndex(p => p.profile_id === profile));
  const overlayFill = kind === "clients" ? "var(--primary)" : profile === "all" ? "var(--highlight)" : `var(--chart-${profileIndex % 5 + 1})`;
  const maxMunicipality = Math.max(1, ...(municipalities ?? []).map(item => item.lookalikes));
  const layer = LAYERS.find(l => l.value === kind) ?? LAYERS[0];
  const counts = density?.cells.map(c => c.n) ?? [];
  const lo = counts.length ? Math.min(...counts) : 0, hi = counts.length ? Math.max(...counts) : 0;
  const top3 = (municipalities ?? []).slice(0, 3);
  const summary = top3.length ? `Οι περισσότερες προτάσεις βρίσκονται: ${top3.map(m => `${short(m.municipality)} (${number(m.lookalikes)})`).join(", ")}.` : "";
  const description = `Χάρτης Ελλάδας: ${layer.label.toLowerCase()}${density ? `, ${number(density.total)} συνολικά` : ""}. ${summary}`;
  const errorState = (message: string) => <div className="space-y-3"><ErrorBanner message={`Δεν φορτώθηκαν τα γεωγραφικά δεδομένα: ${message}`} /><Button variant="outline" onClick={() => setRetry(value => value + 1)}><RotateCcw className="h-4 w-4" />Επανάληψη</Button></div>;

  return <div className="grid items-start gap-6 xl:grid-cols-[2fr_1fr]">
    <Panel className="min-w-0 p-4 md:p-5">
      <div className="mb-4 flex flex-wrap items-end gap-4">
        <div><p id="map-layer-label" className="mb-2 text-sm font-medium">Τι δείχνει ο χάρτης</p>
          <ToggleGroup type="single" value={kind} onValueChange={value => { if (value === "market" || value === "clients" || value === "lookalikes") setKind(value); }} aria-labelledby="map-layer-label" className="flex-wrap justify-start" variant="outline">
            {LAYERS.map(l => <ToggleGroupItem key={l.value} value={l.value} className="px-4">{l.label}</ToggleGroupItem>)}</ToggleGroup></div>
        <div className="w-full max-w-xs"><p id="map-profile-label" className="mb-2 text-sm font-medium">Προφίλ</p><Select value={profile} onValueChange={onProfileChange}><SelectTrigger aria-labelledby="map-profile-label"><SelectValue /></SelectTrigger><SelectContent><SelectItem value="all">Όλα τα προφίλ</SelectItem>{profiles.map(p => <SelectItem key={p.profile_id} value={p.profile_id}>{p.title}</SelectItem>)}</SelectContent></Select></div>
      </div>
      <p className="text-sm text-muted-foreground">{layer.hint}</p>
      {density && <p className="mt-1 text-base font-medium" aria-live="polite">{number(density.total)} {kind === "lookalikes" ? "προτάσεις" : kind === "clients" ? "πελάτες" : "επιχειρήσεις"} στον χάρτη{kind === "lookalikes" && summary ? <span className="block text-sm font-normal text-muted-foreground">{summary}</span> : null}</p>}
      <div className="mt-4">
        {outlineError ? errorState(outlineError) : error ? errorState(error) : !outline || !density ? <div role="status" aria-label="Φόρτωση χάρτη"><Skeleton className="h-[420px] w-full" /></div> : <>
          <BaseMap outline={outline} market={kind === "market" ? density : market} overlay={kind === "market" ? null : density} overlayFill={overlayFill} municipalities={municipalities ?? []} selected={selected} pins={pinsOn ? places : null} place={place} onPlace={setPlace} theme={theme} description={description} />
          <div className="mt-3 flex flex-wrap items-center gap-3 text-sm" aria-label="Υπόμνημα χάρτη">
            <span className="text-muted-foreground">Λίγες</span>
            <span aria-hidden className="h-3 w-40 rounded-full" style={{ background: `linear-gradient(to right, color-mix(in srgb, ${kind === "market" ? "var(--muted-foreground)" : overlayFill} 15%, transparent), ${kind === "market" ? "var(--muted-foreground)" : overlayFill})` }} />
            <span className="text-muted-foreground">Πολλές</span>
            <span className="text-xs text-muted-foreground">Κάθε τετράγωνο ≈ {Math.round(density.cell_deg * 111)} km · από {number(lo)} έως {number(hi)} επιχειρήσεις ανά τετράγωνο · πιο έντονο χρώμα = περισσότερες</span>
          </div>
          {density.cells.length === 0 && <p className="py-4 text-sm text-muted-foreground">Δεν υπάρχουν γεωγραφικά δεδομένα για αυτή την επιλογή.</p>}
        </>}
      </div>
      <details className="mt-4 border-t border-border pt-3"><summary className="cursor-pointer text-sm text-primary">Περισσότερες ρυθμίσεις</summary>
        <div className="mt-3 flex flex-wrap items-end gap-5">
          <div><p className="mb-2 text-xs text-muted-foreground">Μέγεθος τετραγώνων</p><ToggleGroup type="single" value={cell} onValueChange={value => { if (value) setCell(value); }} aria-label="Μέγεθος τετραγώνων" className="flex-wrap justify-start" variant="outline">{RESOLUTIONS.map(r => <ToggleGroupItem key={r.value} value={r.value} className="text-xs">{r.label}</ToggleGroupItem>)}</ToggleGroup></div>
          <div><p className="mb-2 text-xs text-muted-foreground">Εμφάνιση</p><ToggleGroup type="single" value={theme} onValueChange={value => { if (value === "light" || value === "dark") setTheme(value); }} aria-label="Θέμα χάρτη" className="justify-start" variant="outline">{THEMES.map(t => <ToggleGroupItem key={t.value} value={t.value} className="text-xs">{t.label}</ToggleGroupItem>)}</ToggleGroup></div>
          {pinsAvailable && <div><p className="mb-2 text-xs text-muted-foreground">Εμπλουτισμός</p><Button variant={pinsOn ? "default" : "outline"} size="sm" aria-pressed={pinsOn} onClick={() => setPinsOn(value => !value)}><MapPin className="h-4 w-4" />Καρφίτσες Google</Button></div>}
        </div>
      </details>
    </Panel>
    <Panel className="min-w-0 p-4 md:p-5"><h2 className="text-lg font-semibold">Πού να ψάξετε πρώτα</h2>
      <p className="mb-4 mt-1 text-sm text-muted-foreground">Οι δήμοι με τις περισσότερες προτάσεις. Πατήστε έναν δήμο για να τον δείτε στον χάρτη.</p>
      {municipalityError ? errorState(municipalityError) : municipalities === null ? <div role="status" aria-label="Φόρτωση δήμων" className="space-y-6">{[0, 1, 2, 3].map(i => <Skeleton key={i} className="h-12 w-full" />)}</div> : municipalities.length ? <ol className="space-y-1">{municipalities.map((item, index) => <li key={`${item.municipality}-${index}`}><Button variant="ghost" aria-pressed={selected === item} aria-label={`${short(item.municipality)}: ${item.lookalikes} προτάσεις, ${item.market} επιχειρήσεις στην αγορά`} onClick={() => setSelected(selected === item ? null : item)} className={cn("h-auto w-full items-start justify-start gap-3 px-2 py-3 text-left whitespace-normal", selected === item && "bg-accent")}><span className="w-5 shrink-0 font-mono text-xs text-muted-foreground">{index + 1}</span><span className="min-w-0 flex-1"><span className="flex items-start justify-between gap-3"><span className="break-words text-sm font-medium">{short(item.municipality)}</span><span className="shrink-0 font-mono text-primary">{number(item.lookalikes)}</span></span><span className="my-2 block h-2 overflow-hidden rounded-full bg-muted"><span className="block h-full bg-highlight" style={{ width: `${item.lookalikes / maxMunicipality * 100}%` }} /></span><span className="block text-xs text-muted-foreground">{number(item.lookalikes)} προτάσεις · {number(item.market)} επιχειρήσεις στην αγορά</span></span>{selected === item && <MapPin className="h-4 w-4 shrink-0 text-primary" />}</Button></li>)}</ol> : <p className="text-sm text-muted-foreground">Δεν υπάρχουν προτάσεις ανά δήμο.</p>}
    </Panel>
  </div>;
}
