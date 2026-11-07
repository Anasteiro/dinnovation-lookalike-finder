import { useEffect, useRef, useState } from "react";
import { Send, ShieldCheck, Sparkles, Database, Ban } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { api, CHAT_MAX, type ChatAnswer } from "@/lib/api";
import { Panel, ErrorBanner } from "./primitives";

type Turn = { id: number; question: string; answer?: ChatAnswer; error?: string };
const START = ["Ποια προφίλ πελατών βρήκες;", "Δείξε τις 10 καλύτερες παρόμοιες επιχειρήσεις στο icp-2", "Πόσες υποψήφιες ατομικές επιχειρήσεις υπάρχουν στην Αττική;", "Σε ποιους νομούς βρίσκονται οι πελάτες μας;", "Πώς βρίσκεις τις παρόμοιες επιχειρήσεις;"];

function AnswerView({ a, onAsk }: { a: ChatAnswer; onAsk: (q: string) => void }) {
  const refused = a.kind === "refusal";
  return <div className="min-w-0 space-y-3">
    <p className={refused ? "flex gap-2 text-sm text-warning" : "whitespace-pre-line text-sm leading-relaxed"}>{refused && <Ban className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />}{a.text}</p>
    {a.filters && <p className="text-xs text-muted-foreground">Φίλτρα που κατάλαβα: <span className="text-foreground">{a.filters}</span></p>}
    {a.rows.length > 0 && <div className="overflow-x-auto rounded-lg border border-border"><table className="w-full text-left text-xs"><thead className="bg-muted/40 text-muted-foreground"><tr>{a.columns.map(c => <th key={c} scope="col" className="px-3 py-2 font-medium">{c}</th>)}</tr></thead><tbody>{a.rows.map((r, i) => <tr key={i} className="border-t border-border/60">{a.columns.map(c => <td key={c} className="px-3 py-2 align-top">{c === "Επωνυμία" && r["company_id"] ? <button type="button" className="text-left text-primary underline-offset-2 hover:underline" onClick={() => onAsk(String(r["company_id"]))}>{String(r[c] ?? "—")}</button> : String(r[c] ?? "—")}</td>)}</tr>)}</tbody></table></div>}
    {a.ai_summary && <div className="rounded-lg border border-primary/30 bg-primary/5 p-3 text-sm">{a.ai_summary.text ? <><p className="mb-1 flex items-center gap-1.5 text-xs font-medium text-primary"><Sparkles className="h-3.5 w-3.5" aria-hidden />{a.ai_summary.label}</p><p>{a.ai_summary.text}</p></> : <p className="text-xs text-muted-foreground">{a.ai_summary.label}</p>}</div>}
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
      {a.sources.length > 0 && <span className="flex items-center gap-1"><Database className="h-3 w-3" aria-hidden />Πηγή: {a.sources.join(", ")}</span>}
      <span>Απάντηση από κανόνες, χωρίς AI · {Math.round(a.seconds * 1000)} ms</span>
    </div>
    {a.suggestions && <div className="flex flex-wrap gap-2">{a.suggestions.map(s => <Button key={s} size="sm" variant="outline" className="h-auto whitespace-normal py-1.5 text-left text-xs" onClick={() => onAsk(s)}>{s}</Button>)}</div>}
  </div>;
}

export function ChatPanel({ job }: { job: string }) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [ai, setAi] = useState(false);
  const [llm, setLlm] = useState<boolean | null>(null);
  const end = useRef<HTMLDivElement>(null);
  const nextId = useRef(1);
  useEffect(() => { if (turns.length) end.current?.scrollIntoView?.({ block: "end", behavior: "smooth" }); }, [turns]);

  const ask = async (q: string) => {
    const question = q.trim().slice(0, CHAT_MAX);
    if (!question || busy) return;
    const id = nextId.current++;
    setTurns(t => [...t, { id, question }].slice(-30)); setText(""); setBusy(true);
    try {
      const answer = await api.chat(job, question, ai && llm !== false);
      setLlm(answer.llm_available);
      setTurns(t => t.map(x => x.id === id ? { ...x, answer } : x));
    } catch (e) {
      setTurns(t => t.map(x => x.id === id ? { ...x, error: e instanceof Error ? e.message : "Σφάλμα" } : x));
    } finally { setBusy(false); }
  };

  return <div className="grid min-w-0 grid-cols-1 gap-6 xl:grid-cols-[minmax(0,1fr)_300px]">
    <Panel className="flex min-h-[520px] min-w-0 flex-col p-0">
      <div className="flex-1 space-y-5 overflow-y-auto p-5" aria-live="polite">
        {turns.length === 0 && <div className="space-y-3"><p className="text-sm text-muted-foreground">Ρωτήστε κάτι για τα αποτελέσματα. Μερικά παραδείγματα:</p><div className="flex flex-wrap gap-2">{START.map(s => <Button key={s} size="sm" variant="outline" className="h-auto whitespace-normal py-1.5 text-left text-xs" onClick={() => void ask(s)}>{s}</Button>)}</div></div>}
        {turns.map(t => <div key={t.id} className="min-w-0 space-y-3">
          <div className="ml-auto w-fit max-w-[85%] break-words rounded-2xl rounded-br-sm bg-primary px-4 py-2 text-sm text-primary-foreground">{t.question}</div>
          <div className="min-w-0 max-w-full rounded-2xl rounded-bl-sm border border-border bg-card/60 p-4">{t.answer ? <AnswerView a={t.answer} onAsk={q => void ask(q)} /> : t.error ? <ErrorBanner message={t.error} /> : <p className="text-sm text-muted-foreground">{ai ? "Σκέφτομαι… (η σύνοψη με τοπικό AI μπορεί να πάρει έως ένα λεπτό)" : "Σκέφτομαι…"}</p>}</div>
        </div>)}
        <div ref={end} />
      </div>
      <form className="border-t border-border p-4" onSubmit={e => { e.preventDefault(); void ask(text); }}>
        <div className="flex items-end gap-2">
          <Textarea value={text} maxLength={CHAT_MAX} rows={2} placeholder="π.χ. Δείξε 10 παρόμοιες ΙΚΕ στη Θεσσαλονίκη" aria-label="Ερώτηση" className="min-h-[52px] resize-none" onChange={e => setText(e.target.value)} onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); void ask(text); } }} />
          <Button type="submit" size="icon" disabled={busy || !text.trim()} aria-label="Αποστολή"><Send className="h-4 w-4" /></Button>
        </div>
        <div className="mt-2 flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
          <div className="flex items-center gap-2"><Switch id="ai-sum" checked={ai} onCheckedChange={setAi} disabled={llm === false} /><Label htmlFor="ai-sum" className="text-xs font-normal">Σύνοψη με τοπικό AI (πιο αργό){llm === false ? " - δεν υπάρχει τοπικό μοντέλο" : ""}</Label></div>
          <span>{text.length}/{CHAT_MAX}</span>
        </div>
      </form>
    </Panel>
    <Panel className="h-fit space-y-3 p-5 text-sm">
      <p className="flex items-center gap-2 font-medium"><ShieldCheck className="h-4 w-4 text-success" aria-hidden />Πώς προστατεύεται</p>
      <ul className="list-disc space-y-2 pl-4 text-xs text-muted-foreground">
        <li>Τρέχει μόνο σε αυτόν τον υπολογιστή· καμία κλήση στο διαδίκτυο.</li>
        <li>Απαντά μόνο από τα αποτελέσματα της ανάλυσης, με σταθερούς κανόνες ανάγνωσης. Δεν εκτελεί κώδικα ή ερωτήματα βάσης.</li>
        <li>Η ερώτησή σας δεν δίνεται ποτέ σε γλωσσικό μοντέλο, άρα δεν μπορεί να το «ξεγελάσει».</li>
        <li>Η προαιρετική σύνοψη AI βλέπει μόνο τα στοιχεία της απάντησης και απορρίπτεται αν προσθέσει αριθμούς ή συνδέσμους. Σημαίνεται ως κείμενο AI.</li>
        <li>Δεν δίνει προσωπικά στοιχεία ή τζίρο πελατών και δεν κάνει αξιολόγηση πιστοληπτικής ικανότητας.</li>
        <li>Το ιστορικό μένει μόνο σε αυτή τη σελίδα· δεν αποθηκεύεται.</li>
      </ul>
    </Panel>
  </div>;
}
