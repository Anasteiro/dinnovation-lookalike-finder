import { useEffect, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { Database, Gauge, Target, Users2, Download, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { api, subscribeJob, type AnalysisResult, type Health, type JobEvent, type Narration, type ExportFile } from "@/lib/api";
import { watchNarration } from "@/lib/narration";
import { AppShell, type Section } from "@/components/forcecrm/AppShell";
import { UploadPanel, AnalysisProgress } from "@/components/forcecrm/UploadPanel";
import { MetricCard, ErrorBanner, SectionHeader } from "@/components/forcecrm/primitives";
import { ProfileCard } from "@/components/forcecrm/Profiles";
import { LookalikeTable } from "@/components/forcecrm/LookalikeTable";
import { NarrationStatus } from "@/components/forcecrm/NarrationStatus";
import { TransparencyPanel } from "@/components/forcecrm/Transparency";
import { useWorkspace } from "@/components/forcecrm/WorkspaceContext";
import { useWorkspaceNavigation } from "@/components/forcecrm/ExplorationLayout";

export const Route = createFileRoute("/")({
  head: () => ({ meta: [
    { title: "Τοπική ανάλυση πελατών" },
    { name: "description", content: "Ανακαλύψτε προφίλ ιδανικών πελατών και παρόμοιες επιχειρήσεις (τοπική εκτέλεση)." },
    { property: "og:title", content: "Τοπική ανάλυση πελατών" },
    { property: "og:description", content: "Προφίλ πελατών και προτάσεις επιχειρήσεων με πραγματικά δεδομένα από τον τοπικό server." },
    { property: "og:type", content: "website" }, { name: "twitter:card", content: "summary_large_image" },
  ] }), component: Index,
});
const EXPORTS: { file: ExportFile; label: string }[] = [{ file: "profiles.json", label: "Προφίλ JSON" }, { file: "assignments.csv", label: "Αντιστοιχίσεις CSV" }, { file: "lookalikes.csv", label: "Παρόμοιες CSV (ενιαία κατάταξη)" }, { file: "lookalikes_by_profile.csv", label: "Παρόμοιες ανά προφίλ CSV" }, { file: "activity_clarity.csv", label: "Καθαρότητα ΚΑΔ CSV" }];
function Exports({ job }: { job: string }) {
  const [busy, setBusy] = useState<ExportFile | null>(null); const [error, setError] = useState<string | null>(null);
  return <section><h2 className="mb-3 font-semibold">Εξαγωγές</h2><div className="flex flex-wrap gap-3">{EXPORTS.map(({ file, label }) => <Button key={file} variant="outline" disabled={busy !== null} className="h-auto min-h-10 whitespace-normal py-2" onClick={async () => { setBusy(file); setError(null); try { await api.download(job, file); } catch (e) { setError(e instanceof Error ? e.message : String(e)); } finally { setBusy(null); } }}>{busy === file ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}{label}</Button>)}</div>{error && <div className="mt-3"><ErrorBanner message={error} /></div>}</section>;
}
function Index() {
  const { job, setJob, data, setData, section, setSection, profile, setProfile, presenter, setPresenter } = useWorkspace();
  const { go } = useWorkspaceNavigation();
  const [events, setEvents] = useState<JobEvent[]>([]);
  const [busy, setBusy] = useState(false); const [error, setError] = useState<string | null>(null);
  const [ended, setEnded] = useState(false); const [retry, setRetry] = useState(0);
  const [narration, setNarration] = useState<Narration | null>(null); const [narrationError, setNarrationError] = useState<string | null>(null);
  const [narrationTick, setNarrationTick] = useState(0); const [resultsTick, setResultsTick] = useState(0);
  const [health, setHealth] = useState<Health | null>(null); const [narrating, setNarrating] = useState(false);
  const ready = data !== null;
  useEffect(() => {
    if (!job || data) return;
    return subscribeJob(job, {
      event: (event) => setEvents((previous) => [...previous, event]),
      error: setError,
      end: (status) => { if (status === "done") setEnded(true); else setError(`Η ανάλυση τερματίστηκε: ${status}`); },
    });
  }, [job]);
  useEffect(() => {
    if (!job || !ended) return;
    const controller = new AbortController(); setError(null);
    api.results(job, controller.signal).then((result) => {
      if (controller.signal.aborted) return;
      setData(result); setSection("overview");
    }).catch((e: Error) => { if (!controller.signal.aborted) setError(e.message); });
    return () => controller.abort();
  }, [job, ended, retry, resultsTick]);
  useEffect(() => {
    if (!ready) return;
    const controller = new AbortController();
    api.health(controller.signal).then((h) => { if (!controller.signal.aborted) setHealth(h); }).catch(() => {});
    return () => controller.abort();
  }, [ready]);
  useEffect(() => {
    if (!job || !ready) return;
    return watchNarration(job, { update: setNarration, done: () => setResultsTick((v) => v + 1), error: setNarrationError });
  }, [job, ready, narrationTick]);
  const reset = () => { setJob(null); setData(null); setSection("overview"); setEvents([]); setError(null); setEnded(false); setProfile("all"); setNarration(null); setNarrationError(null); setHealth(null); };
  const analyze = async (file: File | null) => {
    setBusy(true); setError(null);
    try { const result = await api.analyze(file); if (!result.job_id) throw new Error("Ο server δεν επέστρεψε αναγνωριστικό ανάλυσης."); setEvents([]); setEnded(false); setJob(result.job_id); }
    catch (e) { setError(e instanceof Error ? e.message : String(e)); }
    finally { setBusy(false); }
  };
  const narrate = async () => {
    if (!job) return;
    setNarrating(true); setNarrationError(null);
    try { await api.narrate(job); setNarrationTick((v) => v + 1); }
    catch (e) { setNarrationError(e instanceof Error ? e.message : String(e)); }
    finally { setNarrating(false); }
  };
  const navigate = go;
  if (!job) return <UploadPanel busy={busy} error={error} onAnalyze={analyze} />;
  if (!data) return <AnalysisProgress events={events} error={error} loadingResults={ended} onRetry={() => setRetry((v) => v + 1)} onBack={reset} />;
  const profiles = data.profiles;
  const showProfiles = <div className="space-y-4"><NarrationStatus narration={narration} error={narrationError} canNarrate={Boolean(health?.llm_model && health?.ollama_running)} busy={narrating} onNarrate={narrate} /><div className="grid items-start gap-5 xl:grid-cols-2">{profiles.map((p) => <ProfileCard key={p.profile_id} p={p} onViewLookalikes={() => { setProfile(p.profile_id); navigate("lookalikes"); }} />)}</div></div>;
  return <AppShell section={section} onNavigate={navigate} onNewUpload={reset} source={data.dataset.source} presenter={presenter} onPresenterChange={setPresenter}>
    {section === "overview" && <><SectionHeader eyebrow="Επισκόπηση" title="Αποτελέσματα ανάλυσης" /><div className="grid grid-cols-2 gap-4 xl:grid-cols-4"><MetricCard icon={Target} label="Προφίλ ICP" value={String(data.dataset.profiles_found)} /><MetricCard icon={Users2} label="Αντιστοιχισμένα ΑΦΜ / ΑΦΜ πελατών" value={`${data.dataset.matched_vats.toLocaleString("el-GR")} / ${data.dataset.client_vats.toLocaleString("el-GR")}`} /><MetricCard icon={Database} label="Μέγεθος αγοράς" value={data.dataset.market_size.toLocaleString("el-GR")} /><MetricCard icon={Gauge} label="Χρόνος εκτέλεσης" value={`${data.dataset.execution_seconds.toLocaleString("el-GR")} s`} /></div><button type="button" onClick={() => navigate("transparency")} className="flex items-center gap-2 text-xs text-muted-foreground transition-colors hover:text-foreground"><span className="h-2 w-2 shrink-0 rounded-full bg-success" />Λειτουργεί 100% τοπικά · καμία κλήση σε εξωτερικό AI</button><Exports job={job} /><section><h2 className="mb-4 text-lg font-semibold">{data.dataset.profiles_found === 1 ? "Ένα ICP" : "Προφίλ ιδανικών πελατών"}</h2>{profiles.length ? showProfiles : <p className="text-sm text-muted-foreground">Δεν βρέθηκαν προφίλ πελατών.</p>}</section></>}
    {section === "profiles" && <><SectionHeader eyebrow="Προφίλ πελατών" title={data.dataset.profiles_found === 1 ? "Ένα ICP" : "Προφίλ ιδανικών πελατών"} />{profiles.length ? showProfiles : <p className="text-muted-foreground">Δεν βρέθηκαν προφίλ πελατών.</p>}</>}
    {section === "lookalikes" && <><SectionHeader eyebrow="Προτάσεις" title="Παρόμοιες επιχειρήσεις" /><LookalikeTable job={job} profiles={profiles} profile={profile} onProfileChange={setProfile} /></>}
    {section === "transparency" && <><SectionHeader eyebrow="Διαφάνεια" title="Διαφάνεια & Ευρωπαϊκός Κανονισμός Τεχνητής Νοημοσύνης" /><TransparencyPanel job={job} /></>}
    
  </AppShell>;
}
