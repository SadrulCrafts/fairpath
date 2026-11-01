"""Preprocessing that lives *inside* the scikit-learn Pipeline.

Because imputation, scaling and one-hot encoding are fitted inside each training fold,
no statistic from a validation or test row can leak into training.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fairpath.config import DatasetSpec


def build_preprocessor(
    spec: DatasetSpec, features: list[str], min_frequency: int | None = None
) -> ColumnTransformer:
    cat = [f for f in features if spec.feature(f).type == "categorical"]
    num = [f for f in features if f not in cat]
    if min_frequency is None:
        # rare categories (e.g. nationalities with 2 students) are pooled into one column
        min_frequency = 10 if spec.name == "dropout" else 5
    transformers = []
    if num:
        transformers.append(
            (
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                num,
            )
        )
    if cat:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="infrequent_if_exist",
                                min_frequency=min_frequency,
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                cat,
            )
        )
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


def output_to_input_map(preprocessor: ColumnTransformer) -> list[str]:
    """For every transformed column, the original feature it came from (for SHAP aggregation)."""
    mapping: list[str] = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "remainder" or trans == "drop":
            continue
        if name == "num":
            mapping.extend(cols)
        else:
            enc: OneHotEncoder = trans.named_steps["onehot"]
            for col, cats, inf in zip(
                cols, enc.categories_, enc.infrequent_categories_ or [None] * len(cols), strict=True
            ):
                n_out = len(cats) - (len(inf) if inf is not None else 0) + (1 if inf is not None else 0)
                mapping.extend([col] * n_out)
    return mapping
