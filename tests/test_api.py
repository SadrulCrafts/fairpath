import io

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(artifacts_dir, tmp_path_factory):
    import os

    reports = tmp_path_factory.mktemp("reports")
    os.environ["FAIRPATH_ARTIFACTS"] = str(artifacts_dir)
    os.environ["FAIRPATH_REPORTS"] = str(reports)
    from fairpath.api.main import app

    with TestClient(app) as c:
        yield c


def test_health_and_meta(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and set(h["models"]) == {"dropout", "student_mat"}
    m = client.get("/api/meta").json()
    assert [d["name"] for d in m["datasets"]] == ["dropout", "student_mat"]
    assert "never" in m["disclaimer"].lower()


def test_schema_and_template(client):
    s = client.get("/api/datasets/dropout/schema").json()
    assert any(f["name"] == "gender" and not f["in_model"] for f in s["fields"])
    t = client.get("/api/datasets/dropout/template.csv")
    assert t.status_code == 200 and t.text.startswith("student_id,")
    assert "gender" not in t.text.splitlines()[0].split(",")


def test_predict_happy_path(client):
    r = client.post("/api/datasets/student_mat/predict", json={"features": {"studytime": 1, "absences": 20}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 <= body["risk"] <= 1 and body["explanation"]["top"] and body["disclaimer"]
    assert r.headers["cache-control"] == "no-store"


def test_predict_validation_errors(client):
    r = client.post("/api/datasets/student_mat/predict", json={"features": {"absences": 500}})
    assert r.status_code == 422 and r.json()["errors"][0]["field"] == "absences"
    r = client.post("/api/datasets/student_mat/predict", json={"features": {"favourite_colour": "blue"}})
    assert r.status_code == 422
    assert client.post("/api/datasets/nope/predict", json={"features": {}}).status_code == 404


def test_whatif_is_light(client):
    r = client.post("/api/datasets/dropout/whatif", json={"features": {"debtor": 1}})
    assert r.status_code == 200 and "counterfactuals" not in r.json()


def test_batch_and_pdf(client, student):
    df = student.X.head(12).copy()
    df.insert(0, "student_id", [f"S{i}" for i in range(12)])
    data = df.to_csv(index=False).encode()
    r = client.post(
        "/api/datasets/student_mat/batch", files={"file": ("c.csv", io.BytesIO(data), "text/csv")}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n"] == 12 and body["rows"][0]["student_id"] == "S0"
    pdf = client.post(
        "/api/datasets/student_mat/batch/report.pdf", files={"file": ("c.csv", io.BytesIO(data), "text/csv")}
    )
    assert pdf.headers["content-type"] == "application/pdf" and pdf.content[:4] == b"%PDF"


def test_batch_schema_mismatch(client, student):
    data = student.X.head(3).drop(columns=["studytime"]).to_csv(index=False).encode()
    r = client.post(
        "/api/datasets/student_mat/batch", files={"file": ("c.csv", io.BytesIO(data), "text/csv")}
    )
    assert r.status_code == 422 and r.json()["errors"][0]["field"] == "studytime"


def test_missing_reports_are_404(client):
    assert client.get("/api/reports/overview").status_code == 404
    assert client.get("/api/reports/dropout/secrets").status_code == 404
