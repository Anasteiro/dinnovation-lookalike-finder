import { createFileRoute } from "@tanstack/react-router";
import { ExplorationLayout } from "@/components/forcecrm/ExplorationLayout";
import { FiltersPanel } from "@/components/forcecrm/FiltersPanel";
import { SectionHeader } from "@/components/forcecrm/primitives";
import { useWorkspace } from "@/components/forcecrm/WorkspaceContext";

export const Route = createFileRoute("/filters")({
  head: () => ({ meta: [
    { title: "Φίλτρα και παράγοντες" },
    { name: "description", content: "Φιλτράρισμα των προτάσεων και πρόσθετοι παράγοντες από ανοικτά δημόσια δεδομένα (τοπική εκτέλεση)." },
  ] }), component: FiltersPage,
});
function FiltersPage() {
  const { job, data, profile, setProfile } = useWorkspace();
  return <ExplorationLayout section="filters"><SectionHeader eyebrow="Ιδανικός πελάτης" title="Φίλτρα και πρόσθετοι παράγοντες" desc="Περιορίστε τη λίστα με τα κριτήριά σας και προσθέστε παράγοντες από ανοικτά δεδομένα (πληθυσμός, ανάπτυξη, ανταγωνισμός)." />{job && data && <FiltersPanel job={job} profiles={data.profiles} profile={profile} onProfileChange={setProfile} />}</ExplorationLayout>;
}
