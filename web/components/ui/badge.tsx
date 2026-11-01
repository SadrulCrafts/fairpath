import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11.5px] font-medium leading-5 [&_svg]:size-3",
  {
    variants: {
      variant: {
        default: "border-transparent bg-secondary text-secondary-foreground",
        outline: "text-muted-foreground",
        brand: "border-transparent bg-brand-soft text-brand",
        good: "border-good/30 bg-good/10 text-[#006300] dark:text-good",
        warn: "border-warn/40 bg-warn/15 text-[#7a5300] dark:text-warn",
        serious: "border-serious/40 bg-serious/15 text-[#9c3d14] dark:text-serious",
        critical: "border-critical/30 bg-critical/10 text-critical",
      },
    },
    defaultVariants: { variant: "default" },
  },
);

export function Badge({ className, variant, ...p }: React.HTMLAttributes<HTMLSpanElement> & VariantProps<typeof badgeVariants>) {
  return <span className={cn(badgeVariants({ variant }), className)} {...p} />;
}
