"""Request / response models for the REST API (these also drive the OpenAPI docs at /docs)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "features": {
                    "course": "9500",
                    "admission_grade": 128.0,
                    "tuition_up_to_date": 0,
                    "debtor": 1,
                    "scholarship_holder": 0,
                    "age_at_enrollment": 19,
                },
                "explain": True,
                "counterfactual": True,
            }
        }
    )

    features: dict[str, Any] = Field(
        ...,
        description="Feature values keyed by feature name. Missing fields "
        "are filled with typical values from the training data.",
    )
    explain: bool = Field(True, description="Include SHAP explanation")
    counterfactual: bool = Field(True, description="Include actionable 'path to lower risk' options")


class Confidence(BaseModel):
    prediction_set: list[int]
    level: Literal["confident", "uncertain", "atypical"]
    explanation: str
    coverage: float


class Contribution(BaseModel):
    feature: str
    label: str
    shap: float
    value: Any
    display_value: str


class Explanation(BaseModel):
    base_value: float
    output_value: float
    top: list[Contribution]
    rest_shap: float
    space: str


class Change(BaseModel):
    feature: str
    label: str
    to_label: str
    from_label: str
    action: str


class CounterfactualOption(BaseModel):
    changes: list[Change]
    new_risk: float
    risk_reduction: float
    n_changes: int


class Counterfactuals(BaseModel):
    current_risk: float
    target_risk: float
    reachable: bool
    options: list[CounterfactualOption]
    actionable_features: list[str]
    message: str | None = None


class PredictResponse(BaseModel):
    dataset: str
    model_version: str
    risk: float = Field(..., description="Calibrated probability of the at-risk outcome")
    band: Literal["low", "moderate", "high"]
    band_text: str
    flagged: bool
    threshold: float
    confidence: Confidence
    fairness_note: dict[str, Any]
    explanation: Explanation | None = None
    counterfactuals: Counterfactuals | None = None
    disclaimer: str


class ValidationIssue(BaseModel):
    row: int | None
    field: str
    message: str


class BatchRow(BaseModel):
    row: int
    student_id: str | None = None
    risk: float
    band: str
    flagged: bool
    confidence: str
    top_factors: list[str]


class BatchResponse(BaseModel):
    dataset: str
    n: int
    summary: dict[str, Any]
    rows: list[BatchRow]
    disclaimer: str


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    models: dict[str, str]
    synthetic: bool
