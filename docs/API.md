# REST API

Interactive docs: `/docs` (Swagger UI) and `/redoc`. OpenAPI schema: `/api/openapi.json`.
`{dataset}` is `dropout` or `student_mat`. The API stores nothing: requests and uploads are processed in memory and discarded.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | liveness + loaded model versions |
| GET | `/api/meta` | datasets, model labels, thresholds, risk bands, disclaimer |
| GET | `/api/datasets/{dataset}/schema` | every input field: type, label, options, ranges, defaults, actionable / protected flags |
| GET | `/api/datasets/{dataset}/examples` | three synthetic example students |
| POST | `/api/datasets/{dataset}/predict` | risk, band, confidence, SHAP explanation, counterfactuals, fairness note |
| POST | `/api/datasets/{dataset}/whatif` | lighter predict (no counterfactuals) for live sliders |
| GET | `/api/datasets/{dataset}/template.csv` | CSV template for batch scoring |
| POST | `/api/datasets/{dataset}/batch` | multipart CSV (≤ 5,000 rows, 2 MB) -> per-row risk, band, confidence, top factors |
| POST | `/api/datasets/{dataset}/batch/report.pdf` | same input -> printable PDF summary |
| GET | `/api/reports/overview` · `/api/reports/thesis` | cross-dataset overview, thesis reproduction |
| GET | `/api/reports/{dataset}/{name}` | `benchmark`, `fairness_study`, `preflight`, `final_model`, `schema`, `examples` |

## Example

```bash
curl -s -X POST http://localhost:8000/api/datasets/dropout/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"course": "9500", "tuition_up_to_date": 0, "debtor": 1, "scholarship_holder": 0}}'
```

```json
{
  "risk": 0.958,
  "band": "high",
  "flagged": true,
  "threshold": 0.37,
  "confidence": {
    "prediction_set": [
      1
    ],
    "level": "confident",
    "coverage": 0.9
  },
  "explanation": {
    "top": [
      {
        "feature": "tuition_up_to_date",
        "label": "Tuition fees up to date",
        "display_value": "No",
        "shap": 3.046
      },
      {
        "feature": "debtor",
        "label": "Has outstanding debt to the institution",
        "display_value": "Yes",
        "shap": 0.978
      }
    ]
  },
  "counterfactuals": {
    "options": [
      {
        "changes": [
          "Set up a payment plan or emergency bursary so tuition is up to date"
        ],
        "new_risk": 0.343
      },
      {
        "changes": [
          "Set up a payment plan or emergency bursary so tuition is up to date",
          "Refer to financial counselling to clear outstanding fees"
        ],
        "new_risk": 0.107
      }
    ]
  }
}
```
(Real response from the v1.0.0 dropout model; `fairness_note`, `band_text` and `disclaimer` omitted for space.)

Missing fields are filled with typical values from the training data. Unknown fields, unknown category codes and out-of-range numbers
return **422** with a list of `{row, field, message}` errors.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `FAIRPATH_ARTIFACTS` | `artifacts` | model bundles (falls back to `artifacts/synthetic`) |
| `FAIRPATH_REPORTS` | `reports` | pre-computed reports |
| `FAIRPATH_WEB_DIR` | `web/out` | static dashboard to serve at `/` |
| `FAIRPATH_CORS_ORIGINS` | `*` | allowed browser origins (comma-separated) |
| `FAIRPATH_LOG_LEVEL` | `INFO` | logging level |
| `PORT` | `7860` (Docker) | port inside the container |
