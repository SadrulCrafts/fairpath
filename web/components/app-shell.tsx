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
import { Badge } from "@/components/ui/badge";

const NAV_GROUPS: { label?: string; items: { href: string; label: string; icon: typeof LayoutDashboard }[] }[] = [
  { items: [{ href: "/", label: "Overview", icon: LayoutDashboard }] },
  {
    label: "Use the model",
    items: [
      { href: "/predict/", label: "Single student", icon: UserRound },
      { href: "/what-if/", label: "What-if simulator", icon: SlidersHorizontal },
      { href: "/batch/", label: "Batch scoring", icon: Table2 },
    ],
  },
  {
    label: "Audit the model",
    items: [
      { href: "/fairness/", label: "Fairness audit", icon: Scale },
      { href: "/models/", label: "Model comparison", icon: BarChart3 },
      { href: "/methodology/", label: "Methodology & model card", icon: BookOpen },
    ],
  },
];
const NAV = NAV_GROUPS.flatMap((g) => g.items);

const REPO_URL = process.env.NEXT_PUBLIC_REPO_URL ?? "https://github.com/SadrulCrafts/fairpath";

function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  const [mounted, setMounted] = React.useState(false);
  React.useEffect(() => setMounted(true), []);
  const dark = mounted && resolvedTheme === "dark";
  return (
    <button
      type="button"
      onClick={() => setTheme(dark ? "light" : "dark")}
      className="inline-flex h-9 w-9 items-center justify-center rounded-lg border bg-card text-muted-foreground transition-colors hover:text-foreground"
      aria-label={dark ? "Switch to light mode" : "Switch to dark mode"}
    >
      {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
    </button>
  );
}

/** Two paths that converge into one: the project's idea in a mark. */
function Mark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden>
      <rect width="32" height="32" rx="9" fill="var(--brand)" />
      <path d="M7 10.5c6 0 8 5.5 15 5.5M7 21.5c6 0 8-5.5 15-5.5" fill="none" stroke="#fff" strokeWidth="2.4" strokeLinecap="round" opacity="0.95" />
      <circle cx="24.5" cy="16" r="2.6" fill="#fff" />
    </svg>
  );
}

function Logo() {
  return (
    <Link href="/" className="flex items-center gap-3 rounded-lg focus-visible:outline-offset-4">
      <Mark className="h-8 w-8 shrink-0" />
      <div className="leading-none">
        <div className="font-display text-[19px] font-semibold tracking-tight">FairPath</div>
        <div className="mt-1 text-[11.5px] text-muted-foreground">Fair early warning</div>
      </div>
    </Link>
  );
}

function isActive(path: string, href: string) {
  return href === "/" ? path === "/" : path.startsWith(href.replace(/\/$/, ""));
}

function Nav({ onNavigate }: { onNavigate?: () => void }) {
  const path = usePathname();
  return (
    <nav className="flex flex-col gap-5" aria-label="Main">
      {NAV_GROUPS.map((g, gi) => (
        <div key={gi} className="flex flex-col gap-0.5">
          {g.label && <div className="mb-1 px-2.5 text-[12px] font-medium text-muted-foreground/80">{g.label}</div>}
          {g.items.map(({ href, label, icon: Icon }) => {
            const active = isActive(path, href);
            return (
              <Link
                key={href}
                href={href}
                onClick={onNavigate}
                aria-current={active ? "page" : undefined}
                className={cn(
                  "group relative flex items-center gap-2.5 rounded-lg px-2.5 py-[7px] text-[13.5px] transition-colors",
                  active ? "bg-brand-soft font-medium text-foreground" : "text-muted-foreground hover:bg-accent hover:text-foreground",
                )}
              >
                {active && <span className="absolute inset-y-1.5 left-0 w-[3px] rounded-full bg-brand" aria-hidden />}
                <Icon className={cn("h-4 w-4 shrink-0", active ? "text-brand" : "opacity-80")} />
                {label}
              </Link>
            );
          })}
        </div>
      ))}
    </nav>
  );
}

function DatasetSwitcher() {
  const { meta, dataset, setDataset } = useDataset();
  if (!meta) return <div className="skeleton h-[58px] w-full" />;
  return (
    <div role="radiogroup" aria-label="Dataset" className="grid grid-cols-2 gap-1 rounded-xl border bg-muted/60 p-1">
      {meta.datasets.map((d) => {
        const on = d.name === dataset;
        return (
          <button
            key={d.name}
            type="button"
            role="radio"
            aria-checked={on}
            onClick={() => setDataset(d.name)}
            className={cn(
              "rounded-lg px-2 py-1.5 text-left transition-colors",
              on ? "bg-card shadow-[0_1px_2px_rgba(20,27,38,0.08)] ring-1 ring-border" : "text-muted-foreground hover:text-foreground",
            )}
          >
            <div className="text-[13px] font-medium leading-5">{d.short_title}</div>
            <div className="tnum text-[11px] leading-4 text-muted-foreground">{d.title.match(/n = [\d,]+/)?.[0] ?? d.model_label}</div>
          </button>
        );
      })}
    </div>
  );
}

function TopBar() {
  const path = usePathname();
  const { meta, dataset } = useDataset();
  const current = NAV.find((n) => isActive(path, n.href));
  const ds = meta?.datasets.find((d) => d.name === dataset);
  return (
    <div className="hidden h-16 items-center justify-between gap-4 px-8 lg:flex">
      <div className="flex min-w-0 items-center gap-2 text-[13px] text-muted-foreground">
        <span className="truncate font-medium text-foreground">{current?.label ?? "FairPath"}</span>
        {ds && (
          <>
            <span aria-hidden className="h-4 w-px bg-border" />
            <span className="truncate">{ds.title}</span>
          </>
        )}
      </div>
      <div className="flex items-center gap-2">
        <a
          href={REPO_URL}
          target="_blank"
          rel="noreferrer"
          className="inline-flex h-9 items-center gap-2 rounded-lg border bg-card px-3 text-[13px] text-muted-foreground transition-colors hover:text-foreground"
        >
          <Github className="h-4 w-4" /> Code
        </a>
        <ThemeToggle />
      </div>
    </div>
  );
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = React.useState(false);
  const { meta, error } = useDataset();
  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[272px_1fr]">
      <aside className="hidden border-r bg-card lg:block">
        <div className="sticky top-0 flex h-screen flex-col gap-7 overflow-y-auto px-4 pb-5 pt-5">
          <Logo />
          <DatasetSwitcher />
          <Nav />
          <div className="mt-auto space-y-3 border-t pt-4 text-[11.5px] leading-5 text-muted-foreground">
            {meta?.synthetic && <Badge variant="warn">Running on synthetic data</Badge>}
            <p>Decision support only. A score suggests who to talk to; it never decides what happens to a student.</p>
            <div className="flex items-center justify-between">
              <a href={REPO_URL} className="inline-flex items-center gap-1.5 hover:text-foreground" target="_blank" rel="noreferrer">
                <Github className="h-3.5 w-3.5" /> Source on GitHub
              </a>
              <span className="tnum">v{meta?.version ?? "1.0.0"}</span>
            </div>
          </div>
        </div>
      </aside>

      <header className="sticky top-0 z-30 flex h-16 items-center justify-between border-b bg-background/85 px-4 backdrop-blur lg:hidden">
        <Logo />
        <div className="flex items-center gap-1.5">
          <ThemeToggle />
          <button className="inline-flex h-9 w-9 items-center justify-center rounded-lg border bg-card" onClick={() => setOpen(!open)} aria-label="Menu" aria-expanded={open}>
            {open ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
          </button>
        </div>
      </header>
      {open && (
        <div className="fixed inset-x-0 top-16 z-20 max-h-[calc(100vh-4rem)] space-y-5 overflow-y-auto border-b bg-background p-4 shadow-lg lg:hidden">
          <DatasetSwitcher />
          <Nav onNavigate={() => setOpen(false)} />
        </div>
      )}

      <div className="min-w-0">
        <TopBar />
        <main className="mx-auto max-w-[1200px] px-4 pb-20 pt-6 sm:px-6 lg:px-8 lg:pt-2">
          {error ? (
            <div className="mt-10 rounded-xl border border-critical/30 bg-critical/5 p-5 text-sm">
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
    <div className="mb-9 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
      <div className="max-w-3xl">
        {eyebrow && <div className="eyebrow mb-3">{eyebrow}</div>}
        <h1 className="font-display text-[30px] font-medium leading-[1.12] tracking-[-0.012em] sm:text-[38px]">{title}</h1>
        {children && <div className="mt-3 max-w-[68ch] text-[15px] leading-[1.65] text-muted-foreground">{children}</div>}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </div>
  );
}
