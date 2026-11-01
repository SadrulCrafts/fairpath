"use client";
import * as React from "react";
import { ChevronDown, Lock } from "lucide-react";
import { Input, Label } from "@/components/ui/misc";
import { Select } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { InfoTip } from "@/components/ui/tooltip";
import type { Field, Schema } from "@/lib/types";
import { cn } from "@/lib/utils";

export type Values = Record<string, string | number>;

export function defaultsFrom(schema: Schema): Values {
  const v: Values = {};
  for (const f of schema.fields) v[f.name] = f.default;
  return v;
}

function Segmented({ field, value, onChange }: { field: Field; value: number; onChange: (v: number) => void }) {
  return (
    <div role="radiogroup" aria-label={field.label} className="grid grid-cols-2 gap-1 rounded-md bg-muted p-1">
      {field.options!.map((o) => (
        <button
          key={String(o.value)}
          type="button"
          role="radio"
          aria-checked={Number(value) === Number(o.value)}
          onClick={() => onChange(Number(o.value))}
          className={cn("rounded px-2 py-1 text-[12.5px] transition-colors", Number(value) === Number(o.value) ? "bg-card font-medium shadow-sm" : "text-muted-foreground hover:text-foreground")}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function FieldInput({ field, value, onChange, slider = false }: { field: Field; value: string | number; onChange: (v: string | number) => void; slider?: boolean }) {
  const id = `f-${field.name}`;
  let control: React.ReactNode;
  if (field.type === "binary") {
    control = <Segmented field={field} value={Number(value)} onChange={onChange} />;
  } else if (field.type === "categorical") {
    control = (
      <Select id={id} ariaLabel={field.label} value={String(value)} onValueChange={onChange}
        options={field.options!.map((o) => ({ value: String(o.value), label: o.label }))} />
    );
  } else if (slider) {
    const lo = field.data_min ?? field.min ?? 0, hi = field.data_max ?? field.max ?? 100;
    control = (
      <div className="flex items-center gap-3">
        <Slider aria-label={field.label} min={lo} max={hi} step={field.step ?? 1} value={[Number(value)]} onValueChange={([v]) => onChange(v)} />
        <span className="w-12 shrink-0 text-right text-[13px] font-medium tnum">{Number(value).toFixed((field.step ?? 1) < 1 ? 1 : 0)}</span>
      </div>
    );
  } else {
    control = (
      <Input id={id} type="number" inputMode="decimal" min={field.min} max={field.max} step={field.step ?? 1} value={value}
        onChange={(e) => onChange(e.target.value === "" ? "" : Number(e.target.value))} aria-describedby={field.help ? `${id}-help` : undefined} />
    );
  }
  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-1.5">
        <Label htmlFor={id} className="text-[12.5px]">{field.label}</Label>
        {field.help && <InfoTip label={`About ${field.label}`}><span id={`${id}-help`}>{field.help}</span></InfoTip>}
        {field.actionable && <span className="rounded bg-brand-soft px-1.5 text-[10.5px] font-medium text-brand">changeable</span>}
      </div>
      {control}
    </div>
  );
}

export function StudentForm({ schema, values, onChange, openGroups = 2 }: { schema: Schema; values: Values; onChange: (v: Values) => void; openGroups?: number }) {
  const fields = schema.fields.filter((f) => f.in_model);
  const hidden = schema.fields.filter((f) => !f.in_model);
  const groups = Array.from(new Set(fields.map((f) => f.group)));
  return (
    <div className="space-y-3">
      {groups.map((g, i) => (
        <details key={g} open={i < openGroups} className="group rounded-lg border bg-card">
          <summary className="flex cursor-pointer list-none items-center justify-between px-4 py-3 text-[13.5px] font-medium">
            {g}
            <ChevronDown className="h-4 w-4 text-muted-foreground transition-transform group-open:rotate-180" />
          </summary>
          <div className="grid gap-4 border-t px-4 py-4 sm:grid-cols-2">
            {fields.filter((f) => f.group === g).map((f) => (
              <FieldInput key={f.name} field={f} value={values[f.name] ?? f.default} onChange={(v) => onChange({ ...values, [f.name]: v })} />
            ))}
          </div>
        </details>
      ))}
      {hidden.length > 0 && (
        <p className="flex items-start gap-2 px-1 text-[12px] leading-5 text-muted-foreground">
          <Lock className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          {hidden.map((f) => f.label).join(", ")} {hidden.length > 1 ? "are" : "is"} deliberately not an input to this model.
        </p>
      )}
    </div>
  );
}
