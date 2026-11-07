import { createFileRoute } from "@tanstack/react-router";
import { ExplorationLayout } from "@/components/forcecrm/ExplorationLayout";
import { ChatPanel } from "@/components/forcecrm/ChatPanel";
import { SectionHeader } from "@/components/forcecrm/primitives";
import { useWorkspace } from "@/components/forcecrm/WorkspaceContext";

export const Route = createFileRoute("/chat")({
  head: () => ({ meta: [
    { title: "Ρώτα τα δεδομένα" },
    { name: "description", content: "Ερωτήσεις σε φυσική γλώσσα πάνω στα αποτελέσματα της ανάλυσης, τοπικά και με ελεγμένες απαντήσεις." },
  ] }), component: ChatPage,
});
function ChatPage() {
  const { job, data } = useWorkspace();
  return <ExplorationLayout section="chat"><SectionHeader eyebrow="Βοηθός" title="Ρώτα τα δεδομένα" desc="Απαντά μόνο από τα αποτελέσματα αυτής της ανάλυσης, τοπικά, χωρίς διαδίκτυο. Κάθε απάντηση δείχνει από ποιο αρχείο προέρχεται." />{job && data && <ChatPanel job={job} />}</ExplorationLayout>;
}
