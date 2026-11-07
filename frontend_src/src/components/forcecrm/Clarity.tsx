import { useEffect, useState } from "react";
import { api, type ClarityCompany, type ClarityOverview } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Panel } from "./primitives";

// Activity Clarity Index: how tidy the declared activity (KAD) profile of a business is, 0-100.
export function clarityTone(level?: string | null) {
  if (level === "Καθαρό") return "border-success/40 bg-success/10 text-success";
  if (level === "Μέτριο") return "border-amber-500/40 bg-amber-500/10 text-amber-500";
  if (level === "Θορυβώδες") return "border-destructive/40 bg-destructive/10 text-destructive";
  return "border-border text-muted-foreground";
}

export function ClarityBadge({ value, level }: { value?: number | null | undefined; level?: string | null | undefined }) {
  if (value === null || value === undefined) return <span className="text-muted-foreground">—</span>;
  return <span className={cn("inline-flex items-center gap-1 whitespace-nowrap rounded-full border px-2 py-0.5 text-xs font-medium", clarityTone(level))} title="Δείκτης καθαρότητας δραστηριότητας (0-100)" aria-label={`Καθαρότητα ΚΑΔ ${value} από 100, ${level ?? ""}`}><span className="font-mono">{value}</span>{level && <span>· {level}</span>}</span>;
}

export function ClarityDetail({ job, companyId }: { job: string; companyId: string | number }) {
  const [d, setD] = useState<ClarityCompany | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    const c = new AbortController(); setD(null); setErr(null);
    api.clarityCompany(job, companyId, c.signal).then(r => { if (!c.signal.aborted) setD(r); }).catch((e: Error) => { if (!c.signal.aborted) setErr(e.message); });
    return () => c.abort();
  }, [job, companyId]);
  if (err) return <p className="text-xs text-muted-foreground">Δεν είναι διαθέσιμος ο δείκτης καθαρότητας.</p>;
  if (!d) return <p role="status" className="text-xs text-muted-foreground">Φόρτωση δείκτη καθαρότητας…</p>;
  return <div className="space-y-2">
    <div className="flex flex-wrap items-center gap-2"><ClarityBadge value={d.clarity} level={d.level} /><span className="text-xs text-muted-foreground">{d.explanation}</span></div>
    <div className="h-2 w-full overflow-hidden rounded-full bg-muted"><div className={cn("h-full rounded-full", d.level === "Καθαρό" ? "bg-success" : d.level === "Μέτριο" ? "bg-amber-500" : "bg-destructive")} style={{ width: `${d.clarity}%` }} /></div>
    <details className="text-xs"><summary className="cursor-pointer text-primary">Δηλωμένοι ΚΑΔ ({d.kads_total.toLocaleString("el-GR")})</summary>
      <ul className="mt-2 max-h-64 space-y-1 overflow-y-auto pr-1">{d.kads.map(k => <li key={k.code + k.type} className="flex gap-2"><span className={cn("mt-0.5 h-2 w-2 shrink-0 rounded-full", k.type === "primary" ? "bg-primary" : k.related === false ? "bg-destructive" : "bg-success")} aria-hidden="true" /><span className="break-words"><span className="font-mono">{k.code}</span> {k.description || "—"}<span className="text-muted-foreground"> · {k.type === "primary" ? "κύριος" : k.related === false ? "άσχετος με την κύρια δραστηριότητα" : "σχετικός"}</span></span></li>)}</ul>
      {d.kads_total > d.kads.length && <p className="mt-1 text-muted-foreground">…και {(d.kads_total - d.kads.length).toLocaleString("el-GR")} ακόμη.</p>}
    </details>
  </div>;
}

export function ClarityOverviewPanel({ job }: { job: string }) {
  const [d, setD] = useState<ClarityOverview | null>(null);
  useEffect(() => {
    const c = new AbortController();
    api.clarityOverview(job, c.signal).then(r => { if (!c.signal.aborted) setD(r); }).catch(() => undefined);
    return () => c.abort();
  }, [job]);
  if (!d) return null;
  const max = Math.max(1, ...d.histogram.counts);
  const total = d.histogram.counts.reduce((a, b) => a + b, 0);
  return <Panel className="p-5"><h2 className="font-semibold">Δείκτης καθαρότητας δραστηριότητας</h2>
    <p className="mt-2 text-sm text-muted-foreground">Πόσο «συμμαζεμένο» είναι το προφίλ δραστηριότητας μιας επιχείρησης: ένας ξεκάθαρος κύριος ΚΑΔ, χωρίς δεκάδες άσχετους δηλωμένους κωδικούς. Κάθε δευτερεύων ΚΑΔ συγκρίνεται με τον κύριο (ίδιος κλάδος ή νοηματική συγγένεια)· όσο περισσότεροι άσχετοι, τόσο χαμηλότερος ο δείκτης (0-100).</p>
    <div className="mt-4 grid gap-3 sm:grid-cols-3">{(["Καθαρό", "Μέτριο", "Θορυβώδες"] as const).map(l => <div key={l} className={cn("rounded-lg border p-3", clarityTone(l))}><p className="text-xs">{l}</p><p className="font-mono text-xl">{((d.levels[l] ?? 0) * 100).toLocaleString("el-GR", { maximumFractionDigits: 1 })}%</p><p className="text-xs opacity-80">της αγοράς</p></div>)}</div>
    <div className="mt-4" role="img" aria-label="Κατανομή του δείκτη καθαρότητας στην αγορά"><div className="flex h-28 items-end gap-1">{d.histogram.counts.map((n, i) => <div key={i} className="flex flex-1 flex-col items-center justify-end gap-1"><span className="font-mono text-[10px] text-muted-foreground">{Math.round(n / Math.max(total, 1) * 100)}%</span><div className="w-full rounded-t bg-primary/70" style={{ height: `${Math.max(2, n / max * 80)}px` }} /></div>)}</div>
      <div className="mt-1 flex gap-1">{d.histogram.counts.map((_, i) => <span key={i} className="flex-1 text-center font-mono text-[10px] text-muted-foreground">{d.histogram.edges[i]}–{d.histogram.edges[i + 1]}</span>)}</div></div>
    <p className="mt-3 text-sm">Μέσος όρος αγοράς: <span className="font-mono">{d.market_mean.toLocaleString("el-GR")}</span>/100</p>
    <p className="mt-2 text-xs text-muted-foreground">{d.note}</p>
  </Panel>;
}
