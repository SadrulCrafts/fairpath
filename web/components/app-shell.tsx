"use client";
import * as React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useTheme } from "next-themes";
import {
  BarChart3, BookOpen, Github, LayoutDashboard, Menu, Moon, Scale, SlidersHorizontal, Sun, Table2, UserRound, X,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useDataset } from "@/lib/dataset-context";
import { Select } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";

const NAV = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/predict/", label: "Single student", icon: UserRound },
  { href: "/what-if/", label: "What-if simulator", icon: SlidersHorizontal },
  { href: "/batch/", label: "Batch scoring", icon: Table2 },
  { href: "/fairness/", label: "Fairness audit", icon: Scale },
  { href: "/models/", label: "Model comparison", icon: BarChart3 },
  { href: "/methodology/", label: "Methodology & model card", icon: BookOpen },
];

const REPO_URL = process.env.NEXT_PUBLIC_REPO_URL ?? "https://github.com/your-username/fairpath";

function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  const [mounted, setMounted] = React.useState(false);
  React.useEffect(() => setMounted(true), []);
  const dark = mounted && resolvedTheme === "dark";
  return (
    <button
      type="button"
      onClick={() => setTheme(dark ? "light" : "dark")}
      className="inline-flex h-8 w-8 items-center justify-center rounded-md text-muted-foreground hover:bg-accent hover:text-foreground"
      aria-label={dark ? "Switch to light mode" : "Switch to dark mode"}
    >
      {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
    </button>
  );
}

function Logo() {
  return (
    <Link href="/" className="flex items-center gap-2.5">
      <svg viewBox="0 0 32 32" className="h-7 w-7" aria-hidden>
        <rect width="32" height="32" rx="8" className="fill-foreground" />
        <path d="M8 22c4-8 8-8 10-5s4 3 6-3" fill="none" stroke="var(--brand)" strokeWidth="2.6" strokeLinecap="round" />
        <circle cx="24" cy="14" r="2.2" className="fill-background" />
      </svg>
      <div className="leading-none">
        <div className="text-[15px] font-semibold tracking-tight">FairPath</div>
        <div className="mt-0.5 text-[11px] text-muted-foreground">Fair early warning</div>
      </div>
    </Link>
  );
}

function Nav({ onNavigate }: { onNavigate?: () => void }) {
  const path = usePathname();
  return (
    <nav className="flex flex-col gap-0.5" aria-label="Main">
      {NAV.map(({ href, label, icon: Icon }) => {
        const active = href === "/" ? path === "/" : path.startsWith(href.replace(/\/$/, ""));
        return (
          <Link
            key={href}
            href={href}
            onClick={onNavigate}
            aria-current={active ? "page" : undefined}
            className={cn(
              "flex items-center gap-2.5 rounded-md px-2.5 py-1.5 text-[13.5px] transition-colors",
              active ? "bg-accent font-medium text-foreground" : "text-muted-foreground hover:bg-accent/60 hover:text-foreground",
            )}
          >
            <Icon className="h-4 w-4" />
            {label}
          </Link>
        );
      })}
    </nav>
  );
}

function DatasetSwitcher() {
  const { meta, dataset, setDataset } = useDataset();
  if (!meta) return <div className="skeleton h-9 w-full" />;
  return (
    <div className="space-y-1.5">
      <div className="eyebrow px-0.5">Dataset</div>
      <Select
        ariaLabel="Choose dataset"
        value={dataset}
        onValueChange={setDataset}
        options={meta.datasets.map((d) => ({ value: d.name, label: d.short_title, hint: d.name === "dropout" ? "n = 3,630" : "n = 395" }))}
      />
    </div>
  );
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = React.useState(false);
  const { meta, error } = useDataset();
  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[248px_1fr]">
      <aside className="hidden border-r bg-card/40 lg:block">
        <div className="sticky top-0 flex h-screen flex-col gap-6 p-4">
          <Logo />
          <DatasetSwitcher />
          <Nav />
          <div className="mt-auto space-y-3 text-[11.5px] leading-5 text-muted-foreground">
            {meta?.synthetic && <Badge variant="warn">Running on synthetic data</Badge>}
            <p>Decision support only. Never use a score to penalise a student.</p>
            <div className="flex items-center justify-between">
              <a href={REPO_URL} className="inline-flex items-center gap-1.5 hover:text-foreground" target="_blank" rel="noreferrer">
                <Github className="h-3.5 w-3.5" /> Source
              </a>
              <span className="tnum">v{meta?.version ?? "1.0.0"}</span>
            </div>
          </div>
        </div>
      </aside>

      <header className="sticky top-0 z-30 flex h-14 items-center justify-between border-b bg-background/85 px-4 backdrop-blur lg:hidden">
        <Logo />
        <div className="flex items-center gap-1">
          <ThemeToggle />
          <button className="inline-flex h-8 w-8 items-center justify-center rounded-md hover:bg-accent" onClick={() => setOpen(!open)} aria-label="Menu" aria-expanded={open}>
            {open ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
          </button>
        </div>
      </header>
      {open && (
        <div className="fixed inset-x-0 top-14 z-20 space-y-5 border-b bg-background p-4 shadow-lg lg:hidden">
          <DatasetSwitcher />
          <Nav onNavigate={() => setOpen(false)} />
        </div>
      )}

      <div className="min-w-0">
        <div className="hidden h-12 items-center justify-end gap-1 px-8 lg:flex">
          <ThemeToggle />
        </div>
        <main className="mx-auto max-w-[1180px] px-4 pb-16 pt-4 sm:px-6 lg:px-8 lg:pt-0">
          {error ? (
            <div className="mt-10 rounded-lg border border-critical/30 bg-critical/5 p-5 text-sm">
              <div className="font-medium">Can&apos;t reach the FairPath API</div>
              <p className="mt-1 text-muted-foreground">{error}. If you are running locally, start it with <code className="rounded bg-muted px-1">make api</code>.</p>
            </div>
          ) : children}
        </main>
      </div>
    </div>
  );
}

export function PageHeader({ eyebrow, title, children, actions }: { eyebrow?: string; title: string; children?: React.ReactNode; actions?: React.ReactNode }) {
  return (
    <div className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
      <div className="max-w-3xl">
        {eyebrow && <div className="eyebrow mb-2">{eyebrow}</div>}
        <h1 className="text-[26px] font-semibold leading-tight tracking-tight sm:text-[30px]">{title}</h1>
        {children && <div className="mt-2.5 text-[15px] leading-6 text-muted-foreground">{children}</div>}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </div>
  );
}
