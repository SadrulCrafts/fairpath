"""How much do the end-of-semester columns inflate dropout AUC?

Runs the same model twice on the real UCI dropout file (Enrolled students removed):
once without the 12 "Curricular units" columns (what FairPath uses) and once with them.

    python scripts/leakage_check.py data/raw/dropout.csv
"""

import sys

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score


def main(path: str) -> None:
    df = pd.read_csv(path, sep=";")
    df.columns = [c.strip() for c in df.columns]
    df = df[df.Target != "Enrolled"]
    y = (df.Target == "Dropout").astype(int)
    leaky = [c for c in df.columns if c.startswith("Curricular units")]
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    for name, X in [
        ("without curricular units", df.drop(columns=leaky + ["Target"])),
        ("with curricular units (leaky)", df.drop(columns=["Target"])),
    ]:
        s = cross_val_score(HistGradientBoostingClassifier(random_state=42), X, y, cv=cv, scoring="roc_auc")
        print(f"{name:32s} AUC {s.mean():.3f} (sd {s.std():.3f})  n={len(df)}  features={X.shape[1]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw/dropout.csv")
