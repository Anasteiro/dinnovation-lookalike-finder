import { createFileRoute } from "@tanstack/react-router";
import { ExplorationLayout } from "@/components/forcecrm/ExplorationLayout";
import { MapPanel } from "@/components/forcecrm/MapPanel";
import { SectionHeader } from "@/components/forcecrm/primitives";
import { useWorkspace } from "@/components/forcecrm/WorkspaceContext";

export const Route = createFileRoute("/map")({
  head: () => ({ meta: [
    { title: "Χάρτης επιχειρήσεων" },
    { name: "description", content: "Γεωγραφική πυκνότητα αγοράς, πελατών και προτάσεων (τοπική εκτέλεση)." },
    { property: "og:title", content: "Χάρτης επιχειρήσεων" },
    { property: "og:description", content: "Εξερευνήστε τις προτάσεις ανά περιοχή και προφίλ, χωρίς εξωτερικούς χάρτες." },
    { property: "og:type", content: "website" }, { name: "twitter:card", content: "summary_large_image" },
  ] }), component: MapPage,
});
function MapPage() {
  const { job, data, profile, setProfile } = useWorkspace();
  return <ExplorationLayout section="map"><SectionHeader title="Χάρτης" />{job && data && <MapPanel job={job} profiles={data.profiles} profile={profile} onProfileChange={setProfile} />}</ExplorationLayout>;
}