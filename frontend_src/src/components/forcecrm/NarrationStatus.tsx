import { Loader2, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { Narration } from "@/lib/api";
import { ErrorBanner } from "./primitives";

export function NarrationStatus({ narration, error, canNarrate, busy, onNarrate }: { narration: Narration | null; error: string | null; canNarrate: boolean; busy: boolean; onNarrate: () => void }) {
  if (!narration && !error) return null;
  return <div className="space-y-3" aria-live="polite">
    {narration?.status === "running" && <p role="status" className="flex items-center gap-2 rounded-lg border border-primary/40 bg-primary/10 p-3 text-sm"><Loader2 className="h-4 w-4 shrink-0 animate-spin text-primary" />Το τοπικό AI (EuroLLM) γράφει τις περιγραφές… {narration.done}/{narration.total}</p>}
    {error && <ErrorBanner message={error} />}
    {narration?.status === "idle" && canNarrate && <Button variant="outline" disabled={busy} onClick={onNarrate}>{busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}Δημιουργία περιγραφών με τοπικό AI</Button>}
  </div>;
}
