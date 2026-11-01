import * as React from "react";
import { AlertTriangle, Info, ShieldCheck } from "lucide-react";
import { cn } from "@/lib/utils";

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn("skeleton", className)} aria-hidden />;
}

export function Callout({ tone = "info", title, children, className }: {
  tone?: "info" | "warn" | "good"; title?: string; children: React.ReactNode; className?: string;
}) {
  const Icon = tone === "warn" ? AlertTriangle : tone === "good" ? ShieldCheck : Info;
  return (
    <div
      role={tone === "warn" ? "alert" : "note"}
      className={cn(
        "flex gap-3 rounded-lg border p-4 text-[13.5px] leading-6",
        tone === "warn" && "border-warn/40 bg-warn/10",
        tone === "good" && "border-good/30 bg-good/5",
        tone === "info" && "bg-muted/50",
        className,
      )}
    >
      <Icon className={cn("mt-0.5 h-4 w-4 shrink-0", tone === "warn" ? "text-[#a36d00] dark:text-warn" : tone === "good" ? "text-good" : "text-muted-foreground")} />
      <div>
        {title && <div className="mb-0.5 font-medium text-foreground">{title}</div>}
        <div className="text-muted-foreground">{children}</div>
      </div>
    </div>
  );
}

export function Separator({ className }: { className?: string }) {
  return <div role="separator" className={cn("h-px w-full bg-border", className)} />;
}

export function Label({ className, ...p }: React.LabelHTMLAttributes<HTMLLabelElement>) {
  return <label className={cn("text-[13px] font-medium leading-none", className)} {...p} />;
}

export function Input({ className, ...p }: React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      className={cn("flex h-9 w-full rounded-md border border-input bg-card px-3 text-sm shadow-sm tnum placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/30", className)}
      {...p}
    />
  );
}

export function EmptyState({ icon: Icon, title, children, className }: {
  icon: React.ComponentType<{ className?: string }>; title: string; children?: React.ReactNode; className?: string;
}) {
  return (
    <div className={cn("flex flex-col items-center justify-center rounded-lg border border-dashed px-6 py-12 text-center", className)}>
      <div className="mb-3 rounded-full bg-muted p-3"><Icon className="h-5 w-5 text-muted-foreground" /></div>
      <div className="text-sm font-medium">{title}</div>
      {children && <div className="mt-1 max-w-sm text-[13px] text-muted-foreground">{children}</div>}
    </div>
  );
}
