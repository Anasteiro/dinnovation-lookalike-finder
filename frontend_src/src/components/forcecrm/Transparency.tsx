import { useEffect, useState } from "react";
import { Bot, RotateCcw, ShieldCheck } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { api, type Transparency } from "@/lib/api";
import { ErrorBanner, Gauge, Panel } from "./primitives";
import { ClarityOverviewPanel } from "./Clarity";

const LABELS: Record<string, string> = {
  k: "k", ari: "Σταθερότητα (ARI)", stability: "Σταθερότητα (ARI)", stability_ari: "Σταθερότητα (ARI)",
  gain: "Κέρδος", min_size: "Ελάχιστο μέγεθος", min_cluster_size: "Ελάχιστο μέγεθος", eligible: "Επιλέξιμο",
  temperature: "Θερμοκρασία", seed: "Σπόρος τυχαιότητας", top_p: "Όριο πιθανότητας", num_predict: "Μέγιστο μήκος παραγωγής",
  name: "Όνομα", title: "Τίτλος", model: "Μοντέλο", dataset: "Σύνολο δεδομένων", source: "Πηγή",
  license: "Άδεια", licence: "Άδεια", url: "Σύνδεσμος", link: "Σύνδεσμος", version: "Έκδοση", type: "Τύπος",
  component: "Στοιχείο", usage: "Χρήση", notes: "Σημειώσεις", note: "Σημείωση", description: "Περιγραφή",
  question: "Ερώτηση", answer: "Απάντηση", article: "Άρθρο", requirement: "Απαίτηση", status: "Κατάσταση",
  compliance: "Συμμόρφωση", evidence: "Τεκμηρίωση", risk: "Κίνδυνος", risk_level: "Επίπεδο κινδύνου",
  loading: "Φόρτωση", loading_market_data: "Φόρτωση δεδομένων αγοράς", matching: "Αντιστοίχιση πελατών",
  matching_clients: "Αντιστοίχιση πελατών", embedding: "Διανυσματική αναπαράσταση", embeddings: "Διανυσματικές αναπαραστάσεις",
  clustering: "Ομαδοποίηση", statistics: "Στατιστικά", scoring: "Βαθμολόγηση", lookalikes: "Παρόμοιες επιχειρήσεις",
  narration: "Δημιουργία περιγραφών", total: "Σύνολο", benchmark: "Έλεγχος ακρίβειας",
};
const label = (key: string) => LABELS[key] ?? key;

function formatValue(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "boolean") return value ? "ναι" : "όχι";
  if (typeof value === "number") return value.toLocaleString("el-GR", { maximumFractionDigits: 3 });
  if (typeof value === "string") return value;
  return JSON.stringify(value);
}

function GenericTable({ rows }: { rows: Record<string, unknown>[] }) {
  const columns = [...new Set(rows.flatMap((row) => Object.keys(row)))];
  if (!columns.length) return null;
  return <div className="overflow-x-auto"><Table><TableHeader><TableRow>{columns.map((c) => <TableHead key={c}>{label(c)}</TableHead>)}</TableRow></TableHeader><TableBody>{rows.map((row, i) => <TableRow key={i}>{columns.map((c) => <TableCell key={c} className="whitespace-normal font-mono text-xs">{formatValue(row[c])}</TableCell>)}</TableRow>)}</TableBody></Table></div>;
}

function FieldList({ items }: { items: Record<string, unknown>[] }) {
  return <ul className="space-y-3">{items.map((item, i) => <li key={i} className="rounded-lg border border-border p-3"><dl className="space-y-1">{Object.entries(item).map(([k, v]) => <div key={k} className="flex flex-wrap gap-2 text-sm"><dt className="shrink-0 font-mono text-xs text-muted-foreground">{label(k)}</dt><dd className="min-w-0 flex-1 break-words">{formatValue(v)}</dd></div>)}</dl></li>)}</ul>;
}

export function TransparencyPanel({ job }: { job: string }) {
  const [data, setData] = useState<Transparency | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    const controller = new AbortController(); setLoading(true); setError(null);
    api.transparency(job, controller.signal).then((result) => { if (!controller.signal.aborted) setData(result); }).catch((e: Error) => { if (!controller.signal.aborted) setError(e.message); }).finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [job, retry]);
  if (loading) return <p role="status" className="py-12 text-center text-sm text-muted-foreground">Φόρτωση στοιχείων διαφάνειας…</p>;
  if (error) return <div className="space-y-3"><ErrorBanner message={error} /><Button variant="outline" onClick={() => setRetry((v) => v + 1)}><RotateCcw className="h-4 w-4" />Επανάληψη</Button></div>;
  if (!data) return null;
  const externalCalls = data.execution?.external_ai_calls;
  const timings = data.execution?.timings_s ? Object.entries(data.execution.timings_s) : [];
  const models = data.models;
  const options = models?.llm_options ? Object.entries(models.llm_options) : [];
  return <div className="space-y-5">
    <Panel className="border-success/40 p-5"><div className="flex items-center gap-2"><ShieldCheck className="h-5 w-5 shrink-0 text-success" /><h2 className="font-semibold">Όλα τοπικά</h2></div>{externalCalls !== undefined && <p className="mt-2 text-sm text-success">{externalCalls.toLocaleString("el-GR")} κλήσεις σε εξωτερική τεχνητή νοημοσύνη</p>}{timings.length > 0 && <div className="mt-3 overflow-x-auto"><Table><TableHeader><TableRow><TableHead>Βήμα</TableHead><TableHead>Χρόνος (δευτ.)</TableHead></TableRow></TableHeader><TableBody>{timings.map(([step, seconds]) => <TableRow key={step}><TableCell className="whitespace-normal">{label(step)}</TableCell><TableCell className="font-mono">{seconds.toLocaleString("el-GR", { maximumFractionDigits: 2 })}</TableCell></TableRow>)}</TableBody></Table></div>}</Panel>
    {models && <Panel className="p-5"><h2 className="font-semibold">Μοντέλα</h2><dl className="mt-3 space-y-2 text-sm">{models.embedding && <div className="flex flex-wrap gap-2"><dt className="text-muted-foreground">Διανυσματικές αναπαραστάσεις</dt><dd className="break-words font-mono">{models.embedding}</dd></div>}{models.llm && <div className="flex flex-wrap gap-2"><dt className="text-muted-foreground">Γλωσσικό μοντέλο</dt><dd className="break-words font-mono">{models.llm}</dd></div>}{models.clustering && <div className="flex flex-wrap gap-2"><dt className="text-muted-foreground">Ομαδοποίηση</dt><dd className="break-words font-mono">{models.clustering}</dd></div>}{models.statistics && <div className="flex flex-wrap gap-2"><dt className="text-muted-foreground">Στατιστικά</dt><dd className="break-words font-mono">{models.statistics}</dd></div>}</dl>{options.length > 0 && <div className="mt-3 flex flex-wrap gap-2">{options.map(([k, v]) => <Badge key={k} variant="secondary" className="max-w-full whitespace-normal font-mono text-xs">{label(k)}: {formatValue(v)}</Badge>)}</div>}</Panel>}
    {data.selection && <Panel className="p-5"><h2 className="font-semibold">Επιλογή αριθμού προφίλ</h2>{data.selection.rule && <p className="mt-2 break-words text-sm text-muted-foreground">{data.selection.rule}</p>}{data.selection.k_table && data.selection.k_table.length > 0 && <div className="mt-3"><GenericTable rows={data.selection.k_table} /></div>}</Panel>}
    {data.profiles && data.profiles.length > 0 && <Panel className="p-5"><h2 className="font-semibold">Έλεγχος κειμένων</h2><div className="mt-4 space-y-5">{data.profiles.map((p, i) => <div key={p.profile_id ?? i} className="border-t border-border pt-4 first:border-0 first:pt-0"><div className="flex flex-wrap items-center gap-2"><h3 className="break-words text-sm font-semibold">{p.title ?? p.profile_id ?? `Προφίλ ${i + 1}`}</h3>{p.description_is_ai !== undefined && <Badge variant={p.description_is_ai ? "outline" : "secondary"} className={p.description_is_ai ? "border-primary/40 text-primary" : undefined}>{p.description_is_ai && <Bot className="mr-1 h-3.5 w-3.5 shrink-0" />}{p.description_is_ai ? "Τεχνητή νοημοσύνη · τοπικό μοντέλο" : "Αυτόματο κείμενο"}</Badge>}</div><div className="mt-3 grid gap-4 sm:grid-cols-2">{p.stability !== undefined && <Gauge label="Σταθερότητα" value={p.stability} />}{p.confidence !== undefined && <Gauge label="Βεβαιότητα αντιστοίχισης" value={p.confidence} />}</div>{(p.description_meta?.verified !== undefined || p.description_meta?.source) && <p className="mt-2 text-xs text-muted-foreground">{p.description_meta?.verified !== undefined && `Επαληθευμένο: ${p.description_meta?.verified ? "ναι" : "όχι"}`}{p.description_meta?.verified !== undefined && p.description_meta?.source && " · "}{p.description_meta?.source && `Πηγή: ${p.description_meta?.source}`}</p>}{p.description_meta?.facts && p.description_meta?.facts.length > 0 && <details className="mt-2"><summary className="cursor-pointer text-xs text-primary">Στοιχεία — {p.description_meta?.facts.length}</summary><ul className="mt-2 space-y-1 text-xs text-muted-foreground">{p.description_meta?.facts.map((f, j) => <li key={f.id ?? j} className="break-words"><span className="font-mono">{f.id ?? j + 1}</span> — {f.text}</li>)}</ul></details>}</div>)}</div></Panel>}
    <ClarityOverviewPanel job={job} />
    {data.licenses && data.licenses.length > 0 && <Panel className="p-5"><h2 className="font-semibold">Άδειες δεδομένων/μοντέλων</h2><div className="mt-3"><FieldList items={data.licenses} /></div></Panel>}
    {data.ai_act && data.ai_act.length > 0 && <Panel className="p-5"><h2 className="font-semibold">Ευρωπαϊκός Κανονισμός για την Τεχνητή Νοημοσύνη</h2><ol className="mt-3 space-y-3">{data.ai_act.map((item, i) => <li key={i} className="rounded-lg border border-border p-3"><p className="break-words text-sm font-semibold"><span className="mr-2 font-mono text-primary">{i + 1}.</span>{typeof item["question"] === "string" ? item["question"] : `Σημείο ${i + 1}`}</p>{typeof item["answer"] === "string" ? <p className="mt-1 break-words text-sm text-muted-foreground">{item["answer"]}</p> : <dl className="mt-1 space-y-1">{Object.entries(item).filter(([k]) => k !== "question").map(([k, v]) => <div key={k} className="flex flex-wrap gap-2 text-sm"><dt className="shrink-0 font-mono text-xs text-muted-foreground">{label(k)}</dt><dd className="min-w-0 flex-1 break-words">{formatValue(v)}</dd></div>)}</dl>}</li>)}</ol></Panel>}
    {data.limits && data.limits.length > 0 && <Panel className="p-5"><h2 className="font-semibold">Όρια</h2><ul className="mt-3 list-disc space-y-1 pl-5 text-sm text-muted-foreground">{data.limits.map((limit, i) => <li key={i} className="break-words">{limit}</li>)}</ul></Panel>}
  </div>;
}
