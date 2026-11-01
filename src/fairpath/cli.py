"""Command-line interface.

fairpath download                     # fetch the UCI datasets into data/raw
fairpath synthesize                   # (re)generate the synthetic look-alike data
fairpath reproduce-thesis             # re-run the original thesis protocol
fairpath benchmark  [-d dropout]      # tune + compare models (5x5 CV)
fairpath fairness   [-d dropout]      # mitigation study (pre / in / post)
fairpath train      [-d dropout]      # fit + evaluate the deployed model, write the bundle
fairpath report                       # overview JSON, figures, docs/RESULTS.md
fairpath all                          # everything above, in order
fairpath preflight data.csv --label y --group sex [--score p]   # stand-alone pre-flight check
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from fairpath.config import load_dataset_spec, load_settings
from fairpath.logging_utils import setup_logging

log = logging.getLogger("fairpath")


def _datasets(args, settings) -> list[str]:
    return [args.dataset] if getattr(args, "dataset", None) else list(settings.datasets)


def cmd_download(args, settings):
    from fairpath.data.download import download_dataset

    for name in _datasets(args, settings):
        download_dataset(name, force=args.force)


def cmd_synthesize(args, settings):
    from fairpath.data.loaders import read_raw
    from fairpath.data.synthetic import make_synthetic

    for name in _datasets(args, settings):
        spec = load_dataset_spec(name)
        raw_path = settings.path("data_raw") / spec.file
        if not raw_path.exists():
            log.error("Need the real %s data to fit the synthetic generator (run `fairpath download`).", name)
            sys.exit(1)
        synth = make_synthetic(read_raw(spec, raw_path), spec, seed=settings.seed)
        out = settings.path("data_synthetic") / spec.file
        out.parent.mkdir(parents=True, exist_ok=True)
        synth.to_csv(out, sep=spec.sep, index=False)
        log.info("wrote %s (%d rows)", out, len(synth))


def cmd_reproduce(args, settings):
    from fairpath.pipeline import thesis_repro

    thesis_repro.run(settings)


def _stage(stage_name):
    def runner(args, settings):
        from fairpath.data import load_dataset
        from fairpath.pipeline import benchmark, fairness_study, train

        stage = {"benchmark": benchmark, "fairness": fairness_study, "train": train}[stage_name]
        for name in _datasets(args, settings):
            ds = load_dataset(name, settings, source=args.source)
            if stage_name == "benchmark":
                stage.run(ds, settings, retune=not args.no_tune)
            else:
                stage.run(ds, settings)

    return runner


def cmd_report(args, settings):
    from fairpath.pipeline import reporting

    reporting.run(settings, synthetic=args.source == "synthetic")


def cmd_all(args, settings):
    if args.source != "synthetic":
        cmd_reproduce(args, settings)
    for st in ("benchmark", "fairness", "train"):
        _stage(st)(args, settings)
    cmd_report(args, settings)


def cmd_preflight(args, settings):
    import pandas as pd

    from fairpath.fairness.preflight import minimum_detectable_gap, preflight_v1, preflight_v2

    df = pd.read_csv(args.csv)
    y = df[args.label].to_numpy()
    g = df[args.group].astype(str).to_numpy()
    pf = settings.preflight
    if args.score:
        s = df[args.score].to_numpy(dtype=float)
        res = {
            "v1": preflight_v1(
                y,
                s,
                g,
                threshold=args.threshold,
                min_n=pf["min_group_n"],
                ratio=pf["noise_floor_ratio"],
                max_cv=pf["max_bootstrap_cv"],
            ),
            "v2": preflight_v2(
                y,
                s,
                g,
                threshold=args.threshold,
                target=pf["target_reduction"],
                alpha=pf["alpha"],
                power=pf["power"],
            ),
        }
    else:  # model-free part only: is there enough data to ever measure a 0.05 change?
        counts = pd.Series(g).value_counts().sort_values()
        mde = minimum_detectable_gap(
            int(counts.iloc[0]), int(counts.iloc[1]), float(y.mean()), pf["alpha"], pf["power"]
        )
        res = {
            "groups": counts.to_dict(),
            "minimum_detectable_gap": round(mde, 4),
            "enough_data": mde <= pf["target_reduction"],
            "note": "Pass --score to run the full v1 and v2 checks on model predictions.",
        }
    print(json.dumps(res, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fairpath", description="Fairness-aware student early-warning pipeline")
    p.add_argument("--config", default=None, help="path to an alternative default.yaml")
    p.add_argument("--log-level", default=None)
    sub = p.add_subparsers(dest="command", required=True)

    def ds_args(sp, source=True):
        sp.add_argument("-d", "--dataset", choices=["dropout", "student_mat"], default=None)
        if source:
            sp.add_argument("--source", choices=["auto", "real", "synthetic"], default="auto")

    sp = sub.add_parser("download", help="download the UCI datasets")
    ds_args(sp, source=False)
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(fn=cmd_download)
    sp = sub.add_parser("synthesize", help="generate synthetic look-alike data")
    ds_args(sp, source=False)
    sp.set_defaults(fn=cmd_synthesize)
    sp = sub.add_parser("reproduce-thesis", help="re-run the original thesis protocol")
    sp.set_defaults(fn=cmd_reproduce, source="auto")
    sp = sub.add_parser("benchmark", help="tune and compare models")
    ds_args(sp)
    sp.add_argument("--no-tune", action="store_true", help="reuse reports/<ds>/tuned_params.json")
    sp.set_defaults(fn=_stage("benchmark"))
    for name, helptext in (
        ("fairness", "mitigation study"),
        ("train", "fit and evaluate the deployed model"),
    ):
        sp = sub.add_parser(name, help=helptext)
        ds_args(sp)
        sp.set_defaults(fn=_stage(name))
    sp = sub.add_parser("report", help="build overview, figures and docs/RESULTS.md")
    sp.add_argument("--source", choices=["auto", "real", "synthetic"], default="auto")
    sp.set_defaults(fn=cmd_report)
    sp = sub.add_parser("all", help="run every stage")
    ds_args(sp)
    sp.add_argument("--no-tune", action="store_true")
    sp.set_defaults(fn=cmd_all)
    sp = sub.add_parser("preflight", help="stand-alone pre-flight check on any CSV")
    sp.add_argument("csv")
    sp.add_argument("--label", required=True, help="binary outcome column (1 = positive)")
    sp.add_argument("--group", required=True, help="sensitive attribute column")
    sp.add_argument("--score", default=None, help="model score / probability column (optional)")
    sp.add_argument("--threshold", type=float, default=0.5)
    sp.set_defaults(fn=cmd_preflight)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    settings = load_settings(args.config)
    args.fn(args, settings)


if __name__ == "__main__":
    main()
