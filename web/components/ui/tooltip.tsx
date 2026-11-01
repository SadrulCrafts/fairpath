"use client";
import * as React from "react";
import * as T from "@radix-ui/react-tooltip";
import { Info } from "lucide-react";
import { cn } from "@/lib/utils";

export const TooltipProvider = T.Provider;

export function Tooltip({ content, children, side = "top" }: { content: React.ReactNode; children: React.ReactNode; side?: "top" | "bottom" | "left" | "right" }) {
  return (
    <T.Root delayDuration={150}>
      <T.Trigger asChild>{children}</T.Trigger>
      <T.Portal>
        <T.Content
          side={side}
          sideOffset={6}
          className="z-50 max-w-xs rounded-md border bg-popover px-3 py-2 text-[12.5px] leading-5 text-popover-foreground shadow-lg animate-fade-in"
        >
          {content}
        </T.Content>
      </T.Portal>
    </T.Root>
  );
}

/** Small (i) icon that explains a metric in plain language. */
export function InfoTip({ children, className, label = "What does this mean?" }: { children: React.ReactNode; className?: string; label?: string }) {
  return (
    <Tooltip content={children}>
      <button type="button" aria-label={label} className={cn("inline-flex text-muted-foreground/70 hover:text-foreground", className)}>
        <Info className="h-3.5 w-3.5" />
      </button>
    </Tooltip>
  );
}
