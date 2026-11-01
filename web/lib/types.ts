export type Band = "low" | "moderate" | "high";

export interface FieldOption { value: string | number; label: string; count?: number }
export interface Field {
  name: string; label: string; help?: string | null; group: string;
  type: "numeric" | "categorical" | "binary"; protected: boolean; in_model: boolean;
  actionable: { direction?: "increase" | "decrease" | null; to?: unknown; action: string } | null;
  options?: FieldOption[]; default: string | number; min?: number; max?: number; step?: number;
  data_min?: number; data_max?: number;
}
export interface Schema { dataset: string; blind: boolean; fields: Field[] }

export interface DatasetMeta {
  name: string; title: string; short_title: string; description: string; model_label: string; version: string;
  threshold: number; bands: { low: number; high: number }; alpha: number; primary_sensitive: string;
  sensitive: Record<string, string>; at_risk_label: string; success_label: string; trained_at: string;
  data_source: string; licence: string; citation: string; source_url: string;
}
export interface Meta { version: string; datasets: DatasetMeta[]; disclaimer: string; synthetic: boolean }

export interface Contribution { feature: string; label: string; shap: number; value: unknown; display_value: string }
export interface Change { feature: string; label: string; from_label: string; to_label: string; action: string }
export interface CfOption { changes: Change[]; new_risk: number; risk_reduction: number; n_changes: number }
export interface GroupRow {
  group: string; label: string; n: number; n_at_risk: number; base_rate: number; flag_rate: number;
  tpr: number | null; fpr: number | null; fnr: number | null; ppv: number | null; accuracy: number;
  mean_score: number | null; reliability: "ok" | "caution" | "unreliable"; ece?: number | null; coverage?: number;
}
export interface Gap { value: number | null; lo: number | null; hi: number | null; boot_sd?: number | null; perm_p?: number; label: string }

export interface Prediction {
  dataset: string; model_version: string; risk: number; band: Band; band_text: string; flagged: boolean; threshold: number;
  confidence: { prediction_set: number[]; level: "confident" | "uncertain" | "atypical"; explanation: string; coverage: number };
  fairness_note: { axis: string; axis_label: string; text: string; groups: GroupRow[]; spd: Gap; eodds: Gap; baseline_spd: Gap; baseline_eodds: Gap };
  explanation?: { base_value: number; output_value: number; top: Contribution[]; rest_shap: number; space: string };
  counterfactuals?: { current_risk: number; target_risk: number; reachable: boolean; options: CfOption[]; actionable_features: string[]; message?: string | null };
  disclaimer: string;
}
export interface Example { name: string; risk: number; record: Record<string, string | number>; synthetic: boolean }

export interface Stat { mean: number; sd: number; lo?: number | null; hi?: number | null }
export interface CI { value: number; lo?: number; hi?: number }
export type ValidationIssue = { row: number | null; field: string; message: string };
