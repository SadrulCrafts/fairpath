"""Feature schema: one JSON-serialisable description of every input field.

The schema is built from the dataset spec *and* the training data (category lists,
sensible ranges and defaults), stored next to the model, and reused by

* the API, to validate incoming requests and CSV uploads, and
* the dashboard, to render the input form without hard-coding anything.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fairpath.config import DatasetSpec


def _occupation_label(code: str, base: dict[str, str]) -> str:
    if code in base:
        return base[code]
    try:
        c = int(code)
    except ValueError:
        return f"Code {code}"
    if 100 <= c < 200:  # detailed ISCO-style codes: 1xy -> major group x
        major = (c // 10) % 10
        major_key = "10" if major == 0 else str(major)
        if major_key in base:
            return f"{base[major_key]} (detailed code {c})"
    return f"Code {code}"


def option_label(spec: DatasetSpec, feature: str, value: Any) -> str:
    f = spec.feature(feature)
    labels = {str(k): v for k, v in spec.labels_for(f).items()}
    key = str(value)
    if key.endswith(".0"):
        key = key[:-2]
    if f.labels_ref == "occupation":
        return _occupation_label(key, labels)
    return labels.get(key, f"Code {key}" if key.lstrip("-").isdigit() else key)


def build_schema(spec: DatasetSpec, X: pd.DataFrame, blind: bool) -> dict[str, Any]:
    in_model = set(spec.model_features(blind))
    fields = []
    for f in spec.features:
        col = X[f.name]
        entry: dict[str, Any] = {
            "name": f.name,
            "label": f.label,
            "help": f.help,
            "group": f.group,
            "type": f.type,
            "protected": f.protected,
            "in_model": f.name in in_model,
            "actionable": (f.actionable.model_dump() if f.is_actionable and f.actionable else None),
        }
        if f.type == "categorical":
            counts = col.value_counts()
            entry["options"] = [
                {"value": str(v), "label": option_label(spec, f.name, v), "count": int(n)}
                for v, n in counts.items()
            ]
            entry["default"] = str(counts.index[0])
        elif f.type == "binary":
            labels = {str(k): v for k, v in spec.labels_for(f).items()}
            vals = sorted(col.dropna().unique().tolist())
            entry["options"] = [
                {"value": int(v), "label": labels.get(str(int(v)), str(int(v)))} for v in vals
            ]
            entry["default"] = int(col.mode().iloc[0])
        else:
            lo = float(f.min) if f.min is not None else float(np.nanmin(col))
            hi = float(f.max) if f.max is not None else float(np.nanmax(col))
            entry.update(
                min=lo,
                max=hi,
                step=float(f.step or 1),
                data_min=float(np.nanmin(col)),
                data_max=float(np.nanmax(col)),
                default=float(np.round(np.nanmedian(col), 2)),
            )
        fields.append(entry)
    return {"dataset": spec.name, "blind": blind, "fields": fields}


class SchemaError(ValueError):
    def __init__(self, errors: list[dict[str, Any]]):
        super().__init__(f"{len(errors)} validation error(s)")
        self.errors = errors


def validate_frame(df: pd.DataFrame, schema: dict[str, Any], fill_defaults: bool = False) -> pd.DataFrame:
    """Coerce and check a frame against the schema. Raises SchemaError with row-level details."""
    errors: list[dict[str, Any]] = []
    out = pd.DataFrame(index=df.index)
    for fld in schema["fields"]:
        name = fld["name"]
        if name not in df.columns:
            if fill_defaults:
                out[name] = fld["default"]
                continue
            errors.append({"row": None, "field": name, "message": "missing column"})
            continue
        col = df[name]
        if fld["type"] == "categorical":
            vals = col.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
            allowed = {o["value"] for o in fld["options"]}
            bad = ~vals.isin(allowed)
            for i in vals.index[bad][:20]:
                errors.append({"row": int(i), "field": name, "message": f"unknown value '{vals[i]}'"})
            out[name] = vals.astype(object)
        else:
            nums = pd.to_numeric(col, errors="coerce")
            for i in nums.index[nums.isna()][:20]:
                errors.append({"row": int(i), "field": name, "message": f"not a number: '{col[i]}'"})
            if fld["type"] == "binary":
                allowed_b = {o["value"] for o in fld["options"]}
                bad = nums.notna() & ~nums.isin(allowed_b)
                for i in nums.index[bad][:20]:
                    errors.append(
                        {"row": int(i), "field": name, "message": f"must be one of {sorted(allowed_b)}"}
                    )
            else:
                bad = nums.notna() & ((nums < fld["min"]) | (nums > fld["max"]))
                for i in nums.index[bad][:20]:
                    errors.append(
                        {
                            "row": int(i),
                            "field": name,
                            "message": f"{nums[i]} outside [{fld['min']}, {fld['max']}]",
                        }
                    )
            out[name] = nums.astype(float)
    if errors:
        raise SchemaError(errors)
    return out
