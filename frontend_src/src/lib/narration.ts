import { api, type Narration, type Paragraph, type Profile } from "./api";

export const AI_BADGE = "AI · τοπικό μοντέλο";
export const AUTO_BADGE = "Αυτόματο κείμενο από στατιστικά (χωρίς AI)";

export function paragraphBadge(paragraph: Paragraph): { label: string; ai: boolean } {
  return paragraph.is_ai ? { label: AI_BADGE, ai: true } : { label: AUTO_BADGE, ai: false };
}

export function profileParagraphs(profile: Profile): Paragraph[] | null {
  const paragraphs = profile.description_meta?.paragraphs;
  return paragraphs && paragraphs.length ? paragraphs : null;
}

export type NarrationHandlers = {
  update: (narration: Narration) => void;
  done: () => void;
  error: (message: string) => void;
};

export function watchNarration(job: string, handlers: NarrationHandlers, intervalMs = 1500): () => void {
  let stopped = false;
  let timer: ReturnType<typeof setTimeout> | null = null;
  const tick = async () => {
    try {
      const status = await api.jobStatus(job);
      if (stopped) return;
      const narration = status.narration;
      if (!narration) return;
      handlers.update(narration);
      if (narration.status === "running") timer = setTimeout(() => void tick(), intervalMs);
      else if (narration.status === "done") handlers.done();
      else if (narration.status === "error") handlers.error(narration.message ?? "Η δημιουργία περιγραφών απέτυχε.");
    } catch (e) {
      if (!stopped) handlers.error(e instanceof Error ? e.message : String(e));
    }
  };
  void tick();
  return () => { stopped = true; if (timer) clearTimeout(timer); };
}
