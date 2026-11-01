import type { Band } from "./types";

export const pct = (v: number | null | undefined, digits = 0) =>
  v == null || Number.isNaN(v) ? "–" : `${(v * 100).toFixed(digits)}%`;
export const num = (v: number | null | undefined, digits = 3) =>
  v == null || Number.isNaN(v) ? "–" : v.toFixed(digits);
export const signed = (v: number, digits = 3) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(digits)}`;
export const ci = (lo?: number | null, hi?: number | null, digits = 3) =>
  lo == null || hi == null ? "" : `${lo.toFixed(digits)} – ${hi.toFixed(digits)}`;
export const pval = (p?: number | null) => (p == null ? "–" : p < 0.001 ? "< 0.001" : p.toFixed(3));

export const bandMeta: Record<Band, { label: string; variant: "good" | "serious" | "critical"; color: string }> = {
  low: { label: "Low risk", variant: "good", color: "var(--status-good)" },
  moderate: { label: "Moderate risk", variant: "serious", color: "var(--status-serious)" },
  high: { label: "High risk", variant: "critical", color: "var(--status-critical)" },
};

export const MODEL_LABELS: Record<string, string> = {
  logistic_regression: "Logistic Regression", random_forest: "Random Forest",
  hist_gradient_boosting: "HistGradientBoosting", lightgbm: "LightGBM", xgboost: "XGBoost",
  dummy_most_frequent: "Dummy (most frequent)", dummy_stratified: "Dummy (stratified)",
};
