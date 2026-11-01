"""Copy the pre-computed reports into web/public/data so the dashboard can show every
audit page without waiting for the API (useful when the API sleeps on a free host).

Live predictions still need the API. Run after `fairpath report`:

    python scripts/export_static_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

from fairpath.api import main as api
from fairpath.config import PROJECT_ROOT
from fairpath.pipeline.common import read_json, to_jsonable

OUT = PROJECT_ROOT / "web" / "public" / "data"


def dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), separators=(",", ":")), encoding="utf-8")


def main() -> None:
    api.load_models()
    reports: Path = api.STATE["reports"]
    dump(OUT / "meta.json", api.meta())
    dump(OUT / "overview.json", read_json(reports / "overview.json"))
    dump(OUT / "thesis.json", api.report_thesis())
    for name in api.STATE["predictors"]:
        for report in sorted(api.REPORT_NAMES):
            if (reports / name / f"{report}.json").exists():
                dump(OUT / name / f"{report}.json", api.report(name, report, compact=True))
    size = sum(p.stat().st_size for p in OUT.rglob("*.json"))
    print(
        f"wrote {len(list(OUT.rglob('*.json')))} files ({size / 1e6:.1f} MB) to {OUT.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
