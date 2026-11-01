/** Recompute group metrics in the browser from pre-computed per-group confusion counts. */
export interface SweepGroup { n: number; tp: number[]; fp: number[]; fn: number[]; tn: number[] }
export interface Sweep { thresholds: number[]; groups: Record<string, SweepGroup> }

export interface GroupAt { group: string; n: number; flagRate: number; tpr: number | null; fpr: number | null; ppv: number | null; tp: number; fp: number; fn: number; tn: number }

export function nearestIndex(ths: number[], t: number) {
  let best = 0;
  for (let i = 1; i < ths.length; i++) if (Math.abs(ths[i] - t) < Math.abs(ths[best] - t)) best = i;
  return best;
}

export function groupsAt(sweep: Sweep, thresholds: Record<string, number>): GroupAt[] {
  return Object.entries(sweep.groups).map(([g, c]) => {
    const i = nearestIndex(sweep.thresholds, thresholds[g]);
    const tp = c.tp[i], fp = c.fp[i], fn = c.fn[i], tn = c.tn[i];
    const pos = tp + fn, neg = fp + tn, flag = tp + fp;
    return { group: g, n: c.n, flagRate: flag / c.n, tpr: pos ? tp / pos : null, fpr: neg ? fp / neg : null, ppv: flag ? tp / flag : null, tp, fp, fn, tn };
  });
}

const range = (xs: (number | null)[]) => {
  const v = xs.filter((x): x is number => x != null);
  return v.length >= 2 ? Math.max(...v) - Math.min(...v) : 0;
};

export function summarise(rows: GroupAt[]) {
  const tp = rows.reduce((a, r) => a + r.tp, 0), fp = rows.reduce((a, r) => a + r.fp, 0), fn = rows.reduce((a, r) => a + r.fn, 0);
  const precision = tp + fp ? tp / (tp + fp) : 0, recall = tp + fn ? tp / (tp + fn) : 0;
  const tprGap = range(rows.map((r) => r.tpr)), fprGap = range(rows.map((r) => r.fpr));
  const flags = rows.map((r) => r.flagRate);
  return {
    spd: range(flags), tprGap, fprGap, eodds: Math.max(tprGap, fprGap), ppvGap: range(rows.map((r) => r.ppv)),
    di: Math.max(...flags) > 0 ? Math.min(...flags) / Math.max(...flags) : 1,
    f1: precision + recall ? (2 * precision * recall) / (precision + recall) : 0, precision, recall,
    flagged: tp + fp,
  };
}
