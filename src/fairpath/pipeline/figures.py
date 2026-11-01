"""Static figures for the README and docs (the dashboard draws its own interactive charts).

Colours follow one validated palette: slot 1 blue for the deployed model, slot 2 orange
for the unmitigated baseline, slot 3 aqua for a third series; recessive grey chrome.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "legend.frameon": False,
        "legend.labelcolor": INK2,
    }
)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def pareto(study: dict[str, Any], axis: str, model: str, title: str, path: Path) -> None:
    """Fairness-accuracy scatter. Points are numbered and named in a key, so labels never collide."""
    blk = study["summary"][axis][model]
    methods = [e for e in blk["methods"] if not e.get("degenerate")]
    degenerate = [e["label"] for e in blk["methods"] if e.get("degenerate")]
    fig, (ax, key) = plt.subplots(1, 2, figsize=(9.6, 4.6), gridspec_kw={"width_ratios": [2.3, 1]})
    front = sorted((e["eodds"]["mean"], e["f1"]["mean"]) for e in methods if e["pareto"])
    ax.plot([p[0] for p in front], [p[1] for p in front], ls="--", lw=1.4, color=MUTED, zorder=1)
    lines = []
    for i, e in enumerate(sorted(methods, key=lambda e: e["eodds"]["mean"]), start=1):
        x, y = e["eodds"]["mean"], e["f1"]["mean"]
        needs = e["needs_sensitive_at_prediction"]
        color = ORANGE if e["method"] == "baseline" else (AQUA if needs else BLUE)
        ax.scatter(x, y, s=150, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(
            str(i), (x, y), ha="center", va="center", fontsize=7.5, color="white", fontweight="bold", zorder=4
        )
        lines.append((i, e["label"], color, e["pareto"]))
    ax.set_xlabel("Equalized-odds gap (lower is fairer)")
    ax.set_ylabel("F1, at-risk class (higher is better)")
    ax.set_title(title)
    xs = [e["eodds"]["mean"] for e in methods]
    ys = [e["f1"]["mean"] for e in methods]
    xr, yr = (max(xs) - min(xs)) or 0.05, (max(ys) - min(ys)) or 0.02
    ax.set_xlim(min(xs) - 0.12 * xr, max(xs) + 0.12 * xr)
    ax.set_ylim(min(ys) - 0.15 * yr, max(ys) + 0.15 * yr)

    key.axis("off")
    key.set_xlim(0, 1)
    key.set_ylim(0, 1)
    yk = 0.97
    for i, label, color, on_front in lines:
        key.scatter(0.04, yk, s=110, color=color, edgecolor=SURFACE)
        key.text(0.04, yk, str(i), ha="center", va="center", fontsize=7, color="white", fontweight="bold")
        key.text(0.11, yk, label + ("  \u25c6" if on_front else ""), va="center", fontsize=8.5, color=INK2)
        yk -= 0.075
    yk -= 0.03
    for color, text in (
        (ORANGE, "baseline"),
        (BLUE, "score never depends on group"),
        (AQUA, "uses the group at prediction"),
        (MUTED, "\u25c6 / dashed: Pareto frontier"),
    ):
        key.text(0.04, yk, "\u25cf" if color != MUTED else "", color=color, fontsize=10, va="center")
        key.text(0.11, yk, text, va="center", fontsize=7.5, color=MUTED)
        yk -= 0.06
    if degenerate:
        key.text(
            0.04,
            yk - 0.03,
            "Not shown (degenerate: flag almost\neveryone or no one): " + ", ".join(degenerate),
            fontsize=7,
            color=MUTED,
            va="top",
            wrap=True,
        )
    _save(fig, path)


def group_bars(final: dict[str, Any], axis: str, title: str, path: Path) -> None:
    fb = final["fairness"][axis]
    base = {r["label"]: r for r in fb["baseline"]["groups"]}
    dep = {r["label"]: r for r in fb["deployed"]["groups"]}
    groups = list(dep)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True)
    for ax, key, name in zip(
        axes, ["flag_rate", "tpr"], ["Flag rate", "Recall (at-risk students caught)"], strict=True
    ):
        x = np.arange(len(groups))
        w = 0.36
        bv = [base[g][key] or 0 for g in groups]
        dv = [dep[g][key] or 0 for g in groups]
        ax.bar(x - w / 2 - 0.01, bv, w, color=ORANGE, label="Unmitigated (aware)")
        ax.bar(x + w / 2 + 0.01, dv, w, color=BLUE, label="Deployed (blind + Reweighing)")
        for xi, v in zip(x - w / 2, bv, strict=True):
            ax.text(xi, v + 0.015, f"{v:.0%}", ha="center", fontsize=8, color=INK2)
        for xi, v in zip(x + w / 2, dv, strict=True):
            ax.text(xi, v + 0.015, f"{v:.0%}", ha="center", fontsize=8, color=INK2)
        ax.set_xticks(x, groups)
        ax.set_ylim(0, 1.05)
        ax.set_title(name, fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(title, x=0.01, y=1.06, ha="left", fontweight="bold", color=INK)
    _save(fig, path)


def calibration(final: dict[str, Any], title: str, path: Path) -> None:
    c = final["calibration"]
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1, ls=":", label="Perfect calibration")
    for rows, color, name in (
        (c["reliability_before"], ORANGE, "Before"),
        (c["reliability_after"], BLUE, "After"),
    ):
        ax.plot(
            [r["mean_pred"] for r in rows],
            [r["frac_pos"] for r in rows],
            marker="o",
            ms=5,
            lw=2,
            color=color,
            label=f"{name} ({c['test_ece_before' if name == 'Before' else 'test_ece_after']:.3f} ECE)",
        )
    ax.set_xlabel("Predicted risk")
    ax.set_ylabel("Observed at-risk rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    _save(fig, path)


def shap_bars(final: dict[str, Any], labels: dict[str, str], title: str, path: Path, k: int = 12) -> None:
    gi = final["global_importance"][:k][::-1]
    fig, ax = plt.subplots(figsize=(6.6, 0.32 * k + 1))
    ax.barh(
        [labels.get(r["feature"], r["feature"]) for r in gi],
        [r["mean_abs_shap"] for r in gi],
        color=BLUE,
        height=0.62,
    )
    ax.set_xlabel("Mean |SHAP value| (log-odds)")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def thesis_scatter(repro: dict[str, Any], path: Path) -> None:
    rows = repro["comparison"]
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    lim = (0, 1)
    ax.plot(lim, lim, color=AXIS, lw=1, ls=":")
    for table, color, name in (
        ("5.1", BLUE, "Table 5.1 (performance)"),
        ("6.1 (|SPD|)", ORANGE, "Table 6.1 (|SPD|)"),
        ("noise floor", AQUA, "Noise floor"),
    ):
        sel = [r for r in rows if r["table"] == table]
        ax.scatter(
            [r["thesis"] for r in sel],
            [r["reproduced"] for r in sel],
            s=36,
            color=color,
            edgecolor=SURFACE,
            linewidth=1.5,
            label=name,
            zorder=3,
        )
    ax.set_xlabel("Value printed in the thesis")
    ax.set_ylabel("Value re-computed by this repo")
    ax.set_title("Re-running the thesis protocol")
    ax.legend(loc="upper left", fontsize=8)
    _save(fig, path)
