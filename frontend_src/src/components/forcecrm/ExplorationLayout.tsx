import { Link, useNavigate } from "@tanstack/react-router";
import type { ReactNode } from "react";
import { Button } from "@/components/ui/button";
import { AppShell, type Section } from "./AppShell";
import { useWorkspace } from "./WorkspaceContext";

export function useWorkspaceNavigation() {
  const navigate = useNavigate();
  const state = useWorkspace();
  const go = (section: Section) => {
    state.setSection(section);
    void navigate({ to: section === "map" ? "/map" : section === "filters" ? "/filters" : section === "chat" ? "/chat" : "/" });
    window.scrollTo({ top: 0 });
  };
  const reset = () => {
    state.setJob(null); state.setData(null); state.setProfile("all"); state.setSection("overview");
    void navigate({ to: "/" });
  };
  return { go, reset };
}

export function ExplorationLayout({ section, children }: { section: "map" | "filters" | "chat"; children: ReactNode }) {
  const { job, data, presenter, setPresenter } = useWorkspace();
  const { go, reset } = useWorkspaceNavigation();
  if (!job || !data) return <main className="mx-auto max-w-xl space-y-5 p-8"><h1 className="text-2xl font-semibold">{section === "map" ? "Χάρτης" : section === "chat" ? "Ρώτα τα δεδομένα" : "Φίλτρα και παράγοντες"}</h1><p className="text-muted-foreground">Δεν υπάρχει ενεργή ανάλυση.</p><Button asChild><Link to="/">Νέα ανάλυση</Link></Button></main>;
  return <AppShell section={section} source={data.dataset.source} onNavigate={go} onNewUpload={reset} presenter={presenter} onPresenterChange={setPresenter}>{children}</AppShell>;
}