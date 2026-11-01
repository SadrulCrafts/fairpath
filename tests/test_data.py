import numpy as np
import pandas as pd
import pytest

from fairpath.config import load_dataset_spec, load_settings
from fairpath.data import load_dataset
from fairpath.data.loaders import clean
from fairpath.data.schema import SchemaError, build_schema, option_label, validate_frame


def test_synthetic_has_same_schema_as_spec(student, dropout):
    for ds in (student, dropout):
        assert list(ds.X.columns) == ds.spec.feature_names
        assert ds.is_synthetic
        assert set(np.unique(ds.y)) == {0, 1}


def test_dropout_cleaning_removes_enrolled_and_leakage(dropout):
    assert not any(c.startswith("curricular") for c in dropout.X.columns)
    assert any("Enrolled" in n for n in dropout.notes)
    assert any("leakage" in n for n in dropout.notes)


@pytest.mark.realdata
def test_real_dropout_matches_thesis_counts():
    s = load_settings()
    spec = load_dataset_spec("dropout")
    if not (s.path("data_raw") / spec.file).exists():
        pytest.skip("real data not downloaded")
    ds = load_dataset("dropout", s, source="real")
    assert len(ds.y) == 3630
    assert int(ds.y.sum()) == 1421  # dropouts (thesis section 4.2.2)
    assert ds.X.shape[1] == 24
    assert ds.sensitive["gender"].value_counts().to_dict() == {0: 2381, 1: 1249}


def test_student_target_is_g3_below_10():
    spec = load_dataset_spec("student_mat")
    raw = pd.DataFrame({f.source: ["x"] * 3 if f.type == "categorical" else [1, 2, 3] for f in spec.features})
    raw["G1"], raw["G2"], raw["G3"] = [5, 10, 15], [5, 10, 15], [9, 10, 15]
    X, y, notes = clean(spec, raw)
    assert y.tolist() == [1, 0, 0]
    assert "G1" not in X.columns and "G2" not in X.columns


def test_derived_sensitive_attributes(dropout, student):
    ages = dropout.X["age_at_enrollment"]
    ag = dropout.sensitive["age_group"]
    assert (ag[ages <= 20] == 0).all() and (ag[ages > 25] == 2).all()
    pe = student.sensitive["parental_education"]
    hi = student.X[["Medu", "Fedu"]].max(axis=1) > 2
    assert (pe == hi.astype(int)).all()


def test_validate_frame_reports_problems(student):
    schema = build_schema(student.spec, student.X, blind=True)
    ok = validate_frame(student.X.head(3), schema)
    assert len(ok) == 3
    bad = student.X.head(2).copy()
    bad["age"] = [16, 99]
    bad["school"] = ["GP", "Nowhere High"]
    with pytest.raises(SchemaError) as e:
        validate_frame(bad, schema)
    fields = {err["field"] for err in e.value.errors}
    assert {"age", "school"} <= fields
    with pytest.raises(SchemaError):
        validate_frame(student.X.drop(columns=["studytime"]).head(1), schema)
    filled = validate_frame(student.X.drop(columns=["studytime"]).head(1), schema, fill_defaults=True)
    assert "studytime" in filled.columns


def test_schema_marks_blind_and_actionable(dropout):
    schema = build_schema(dropout.spec, dropout.X, blind=True)
    f = {x["name"]: x for x in schema["fields"]}
    assert f["gender"]["in_model"] is False
    assert f["tuition_up_to_date"]["actionable"] is not None
    assert f["age_at_enrollment"]["actionable"] is None  # protected, never actionable


def test_occupation_labels_fold_detailed_codes():
    spec = load_dataset_spec("dropout")
    assert option_label(spec, "mothers_occupation", 4) == "Administrative staff"
    assert option_label(spec, "mothers_occupation", "122").startswith("Intellectual and scientific")
    assert option_label(spec, "course", "9500") == "Nursing"
