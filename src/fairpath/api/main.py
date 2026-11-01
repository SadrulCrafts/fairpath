"""FastAPI service: predictions, explanations, batch scoring and the pre-computed audit reports.

Privacy by design: the API is stateless. Request bodies and uploaded files are processed
in memory and discarded; nothing a user sends is written to disk or logged.

Run locally:  uvicorn fairpath.api.main:app --reload
"""

from __future__ import annotations

import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from fairpath import __version__
from fairpath.api.pdf import batch_report_pdf
from fairpath.api.schemas import BatchResponse, Health, PredictRequest, PredictResponse
from fairpath.config import PROJECT_ROOT, load_dataset_spec
from fairpath.data.schema import SchemaError
from fairpath.inference import Predictor
from fairpath.logging_utils import setup_logging
from fairpath.pipeline.common import read_json, to_jsonable

log = logging.getLogger("fairpath.api")

DISCLAIMER = (
    "Decision support only. This score estimates risk from historical data and can be wrong, especially for "
    "students unlike those in the training data. It must never be used to penalise, exclude or label a "
    "student. A person who knows the student should review every flag and decide what support to offer."
)
MAX_UPLOAD_BYTES = 2_000_000
MAX_BATCH_ROWS = 5000
REPORT_NAMES = {"benchmark", "fairness_study", "preflight", "final_model", "schema", "examples"}


def _dirs() -> tuple[Path, Path, bool]:
    art = Path(os.getenv("FAIRPATH_ARTIFACTS", PROJECT_ROOT / "artifacts"))
    rep = Path(os.getenv("FAIRPATH_REPORTS", PROJECT_ROOT / "reports"))
    synthetic = False
    if not any(art.glob("*_model.joblib")) and any((art / "synthetic").glob("*_model.joblib")):
        art, rep, synthetic = art / "synthetic", rep / "synthetic", True
    return art, rep, synthetic


STATE: dict[str, Any] = {"predictors": {}, "reports": None, "synthetic": False}


def load_models() -> None:
    art, rep, synthetic = _dirs()
    STATE.update(predictors={}, reports=rep, synthetic=synthetic)
    for path in sorted(art.glob("*_model.joblib")):
        name = path.name.replace("_model.joblib", "")
        try:
            STATE["predictors"][name] = Predictor.from_path(path)
            log.info("loaded model %s from %s", name, path)
        except Exception as exc:  # pragma: no cover - surfaced through /api/health
            log.error("could not load %s: %s", path, exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    load_models()
    yield


app = FastAPI(
    title="FairPath API",
    version=__version__,
    description=(
        "Fairness-aware early warning for student success. Every prediction comes with an explanation, "
        "a confidence indicator, actionable support options and the model's audited fairness record. "
        + DISCLAIMER
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/api/openapi.json",
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("FAIRPATH_CORS_ORIGINS", "*").split(",")],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def no_store(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/api/datasets"):
        response.headers["Cache-Control"] = "no-store"
    return response


def predictor(dataset: str) -> Predictor:
    p = STATE["predictors"].get(dataset)
    if p is None:
        raise HTTPException(
            404, f"No model for dataset '{dataset}'. Available: {sorted(STATE['predictors'])}"
        )
    return p


@app.get("/api/health", response_model=Health, tags=["system"])
def health() -> Health:
    models = {k: v.b.version for k, v in STATE["predictors"].items()}
    return Health(
        status="ok" if models else "degraded",
        version=__version__,
        models=models,
        synthetic=STATE["synthetic"],
    )


@app.get("/api/meta", tags=["system"])
def meta() -> dict[str, Any]:
    out = []
    for name, p in STATE["predictors"].items():
        spec = load_dataset_spec(name)
        out.append(
            {
                "name": name,
                "title": spec.title,
                "short_title": spec.short_title,
                "description": spec.description,
                "model_label": p.b.metadata.get("model_label", p.b.model_name),
                "version": p.b.version,
                "threshold": p.b.threshold,
                "bands": p.b.bands,
                "alpha": p.b.alpha,
                "primary_sensitive": spec.primary_sensitive,
                "sensitive": {k: v.label for k, v in spec.sensitive.items()},
                "at_risk_label": spec.target.at_risk_label,
                "success_label": spec.target.success_label,
                "trained_at": p.b.metadata.get("generated_at"),
                "data_source": p.b.metadata.get("data_source"),
                "licence": spec.licence,
                "citation": spec.citation,
                "source_url": spec.source_url,
            }
        )
    order = {"dropout": 0, "student_mat": 1}
    out.sort(key=lambda d: order.get(d["name"], 9))
    return {
        "version": __version__,
        "datasets": out,
        "disclaimer": DISCLAIMER,
        "synthetic": STATE["synthetic"],
    }


@app.get("/api/datasets/{dataset}/schema", tags=["prediction"])
def schema(dataset: str) -> dict[str, Any]:
    return predictor(dataset).b.schema


@app.get("/api/datasets/{dataset}/examples", tags=["prediction"])
def examples(dataset: str) -> list[dict[str, Any]]:
    path = STATE["reports"] / dataset / "examples.json"
    if not path.exists():
        return []
    return read_json(path)


def _validation_error(exc: SchemaError) -> JSONResponse:
    return JSONResponse(
        status_code=422, content={"detail": "Input did not match the schema", "errors": exc.errors}
    )


@app.post("/api/datasets/{dataset}/predict", response_model=PredictResponse, tags=["prediction"])
def predict(dataset: str, req: PredictRequest):
    p = predictor(dataset)
    unknown = sorted(set(req.features) - {f["name"] for f in p.b.schema["fields"]})
    if unknown:
        return JSONResponse(
            status_code=422,
            content={
                "detail": f"Unknown fields: {unknown}",
                "errors": [{"row": None, "field": f, "message": "unknown field"} for f in unknown],
            },
        )
    try:
        out = p.predict_one(req.features, explain=req.explain, counterfactual=req.counterfactual)
    except SchemaError as exc:
        return _validation_error(exc)
    out["disclaimer"] = DISCLAIMER
    return to_jsonable(out)


@app.post("/api/datasets/{dataset}/whatif", tags=["prediction"])
def whatif(dataset: str, req: PredictRequest) -> Any:
    """Fast path for the what-if simulator: risk, band, confidence and top factors only."""
    p = predictor(dataset)
    try:
        out = p.predict_one(req.features, explain=req.explain, counterfactual=False)
    except SchemaError as exc:
        return _validation_error(exc)
    if "explanation" in out:
        out["explanation"]["all"] = []
    return to_jsonable(out)


@app.get("/api/datasets/{dataset}/template.csv", tags=["batch"])
def template(dataset: str) -> Response:
    p = predictor(dataset)
    fields = [f for f in p.b.schema["fields"] if f["in_model"]]
    header = ["student_id"] + [f["name"] for f in fields]
    example = ["S001"] + [str(f["default"]) for f in fields]
    csv = ",".join(header) + "\n" + ",".join(example) + "\n"
    return Response(
        csv,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="fairpath_{dataset}_template.csv"'},
    )


async def _read_upload(file: UploadFile) -> pd.DataFrame:
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large (limit {MAX_UPLOAD_BYTES // 1_000_000} MB)")
    try:
        text = raw.decode("utf-8-sig")
        sep = ";" if text.splitlines()[0].count(";") > text.splitlines()[0].count(",") else ","
        df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False)
    except Exception as exc:
        raise HTTPException(400, f"Could not read the CSV: {exc}") from exc
    if df.empty:
        raise HTTPException(400, "The CSV has no rows")
    if len(df) > MAX_BATCH_ROWS:
        raise HTTPException(413, f"Too many rows ({len(df)}); the limit is {MAX_BATCH_ROWS}")
    return df


def _score_batch(p: Predictor, df: pd.DataFrame) -> dict[str, Any]:
    ids = None
    for col in ("student_id", "id", "ID"):
        if col in df.columns:
            ids = df[col].astype(str).tolist()
            break
    model_fields = [f["name"] for f in p.b.schema["fields"] if f["in_model"]]
    missing = [c for c in model_fields if c not in df.columns]
    if missing:
        raise SchemaError([{"row": None, "field": c, "message": "missing column"} for c in missing])
    X = p.prepare(df.replace("", None), fill_defaults=True)
    table = p.predict_table(X)
    sv = p.explainer.explain(X)
    labels = {f.name: f.label for f in p.spec.features}
    rows = []
    for i in range(len(X)):
        top = sv.iloc[i].sort_values(ascending=False)
        factors = [f"{labels[f]}: {p.display_value(f, X.iloc[i][f])}" for f, v in top.items() if v > 0][:3]
        rows.append(
            {
                "row": i,
                "student_id": ids[i] if ids else None,
                "risk": float(table.risk[i]),
                "band": table.band[i],
                "flagged": bool(table.flagged[i]),
                "confidence": table.confidence[i],
                "top_factors": factors,
            }
        )
    summary = {
        "bands": table.band.value_counts().to_dict(),
        "flagged": int(table.flagged.sum()),
        "uncertain": int((table.confidence == "uncertain").sum()),
        "mean_risk": float(table.risk.mean()),
        "threshold": p.b.threshold,
    }
    return {"dataset": p.b.dataset, "n": len(X), "summary": summary, "rows": rows, "disclaimer": DISCLAIMER}


@app.post("/api/datasets/{dataset}/batch", response_model=BatchResponse, tags=["batch"])
async def batch(dataset: str, file: UploadFile = File(...)):
    p = predictor(dataset)
    df = await _read_upload(file)
    try:
        return to_jsonable(_score_batch(p, df))
    except SchemaError as exc:
        return _validation_error(exc)


@app.post("/api/datasets/{dataset}/batch/report.pdf", tags=["batch"])
async def batch_pdf(dataset: str, file: UploadFile = File(...)):
    p = predictor(dataset)
    df = await _read_upload(file)
    try:
        result = _score_batch(p, df)
    except SchemaError as exc:
        return _validation_error(exc)
    spec = load_dataset_spec(dataset)
    pdf = batch_report_pdf(result, spec.title, p.b.metadata.get("model_label", p.b.model_name), DISCLAIMER)
    return Response(
        pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="fairpath_{dataset}_report.pdf"'},
    )


@app.get("/api/reports/overview", tags=["reports"])
def report_overview() -> Any:
    path = STATE["reports"] / "overview.json"
    if not path.exists():
        raise HTTPException(404, "overview.json not found: run `fairpath report`")
    return read_json(path)


@app.get("/api/reports/thesis", tags=["reports"])
def report_thesis() -> Any:
    path = STATE["reports"] / "thesis_reproduction.json"
    if not path.exists():
        raise HTTPException(404, "thesis_reproduction.json not found")
    data = read_json(path)
    data.pop("results", None)
    return data


@app.get("/api/reports/{dataset}/{name}", tags=["reports"])
def report(dataset: str, name: str, compact: bool = Query(True, description="drop per-fold rows")) -> Any:
    if name not in REPORT_NAMES:
        raise HTTPException(404, f"Unknown report '{name}'")
    path = STATE["reports"] / dataset / f"{name}.json"
    if not path.exists():
        raise HTTPException(404, f"{path.name} not found for {dataset}")
    data = read_json(path)
    if compact and isinstance(data, dict):
        data.pop("folds", None)
    return data


# The dashboard (Next.js static export) is served from the same origin when it has been built.
_web = Path(os.getenv("FAIRPATH_WEB_DIR", PROJECT_ROOT / "web" / "out"))
if _web.exists():
    app.mount("/", StaticFiles(directory=_web, html=True), name="web")
