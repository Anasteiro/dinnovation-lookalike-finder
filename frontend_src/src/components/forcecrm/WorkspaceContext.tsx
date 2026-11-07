import { createContext, useContext, useState, type ReactNode } from "react";
import type { AnalysisResult } from "@/lib/api";
import type { Section } from "./AppShell";

function useWorkspaceState() {
  const [job, setJob] = useState<string | null>(null);
  const [data, setData] = useState<AnalysisResult | null>(null);
  const [section, setSection] = useState<Section>("overview");
  const [profile, setProfile] = useState("all");
  const [presenter, setPresenter] = useState(false);
  return { job, setJob, data, setData, section, setSection, profile, setProfile, presenter, setPresenter };
}
const WorkspaceContext = createContext<ReturnType<typeof useWorkspaceState> | null>(null);
export function WorkspaceProvider({ children }: { children: ReactNode }) {
  const state = useWorkspaceState();
  return <WorkspaceContext.Provider value={state}>{children}</WorkspaceContext.Provider>;
}
export function useWorkspace() {
  const state = useContext(WorkspaceContext);
  if (!state) throw new Error("WorkspaceProvider is required");
  return state;
}