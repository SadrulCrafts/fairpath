"use client";
import { ThemeProvider } from "next-themes";
import { TooltipProvider } from "@/components/ui/tooltip";
import { DatasetProvider } from "@/lib/dataset-context";
import { AppShell } from "@/components/app-shell";

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <ThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange>
      <TooltipProvider>
        <DatasetProvider>
          <AppShell>{children}</AppShell>
        </DatasetProvider>
      </TooltipProvider>
    </ThemeProvider>
  );
}
