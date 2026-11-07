import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import { percent } from "@/lib/api";

export const number = (n: number) => n.toLocaleString("el-GR");

export function Panel({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("surface rounded-xl border border-border", className)} {...props} />;
}
export function SectionHeader({ eyebrow, title, desc, action }: { eyebrow?: string; title: string; desc?: string; action?: React.ReactNode }) {
  return <div className="flex flex-wrap items-end justify-between gap-4"><div>{eyebrow && <p className="font-mono text-xs uppercase text-primary">{eyebrow}</p>}<h1 className="mt-1 text-2xl font-semibold md:text-3xl">{title}</h1>{desc && <p className="mt-2 text-sm text-muted-foreground">{desc}</p>}</div>{action}</div>;
}
export function MetricCard({ icon: Icon, label, value }: { icon: LucideIcon; label: string; value: string }) {
  return <Panel className="p-4"><div className="flex items-center justify-between gap-2 text-muted-foreground"><span className="text-xs font-medium">{label}</span><Icon className="h-4 w-4 shrink-0" aria-hidden /></div><p className="mt-3 break-words font-mono text-2xl font-semibold tabular-nums">{value}</p></Panel>;
}
export function Gauge({ label, value }: { label: string; value: number }) {
  return <div><div className="mb-2 flex justify-between gap-2 text-xs"><span className="text-muted-foreground">{label}</span><span className="font-mono">{percent(value)}</span></div><div role="meter" aria-label={label} aria-valuenow={value * 100} aria-valuemin={0} aria-valuemax={100} className="h-1.5 overflow-hidden rounded-full bg-muted"><div className="h-full bg-primary" style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} /></div></div>;
}
export function ErrorBanner({ message }: { message: string }) {
  return <p role="alert" className="break-words rounded-lg border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">{message}</p>;
}
