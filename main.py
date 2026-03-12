"""
Charola Checker — Backend
Recibe muestras de corrección de todos los dispositivos,
calcula umbrales adaptativos y los devuelve como modelo JSON.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import sqlite3, json, math, os
from datetime import datetime

app = FastAPI(title="Charola Checker API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = os.environ.get("DB_PATH", "charola.db")

# ── DB setup ────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS samples (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                brightness  REAL    NOT NULL,
                variance    REAL    NOT NULL,
                edge        REAL    NOT NULL,
                label       TEXT    NOT NULL CHECK(label IN ('alive','dead')),
                device_id   TEXT,
                created_at  TEXT    DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_cache (
                id          INTEGER PRIMARY KEY,
                payload     TEXT    NOT NULL,
                sample_count INTEGER NOT NULL,
                updated_at  TEXT    DEFAULT (datetime('now'))
            )
        """)
        conn.commit()

init_db()

# ── Schemas ─────────────────────────────────────────────────────────────────
class Sample(BaseModel):
    brightness:  float
    variance:    float
    edge:        float
    label:       str          # "alive" | "dead"
    device_id:   Optional[str] = None

class SampleBatch(BaseModel):
    samples: List[Sample]

class ModelResponse(BaseModel):
    version:          int
    sample_count:     int
    brightness_thresh: float
    variance_thresh:   float
    edge_thresh:       float
    accuracy:         Optional[float]
    updated_at:       str
    status:           str      # "basic" | "trained"

# ── Helpers ──────────────────────────────────────────────────────────────────
def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0

def compute_model(conn) -> dict:
    rows = conn.execute("SELECT brightness, variance, edge, label FROM samples").fetchall()
    alive = [r for r in rows if r["label"] == "alive"]
    dead  = [r for r in rows if r["label"] == "dead"]

    total = len(rows)

    if len(alive) < 10 or len(dead) < 10:
        return {
            "version": 1,
            "sample_count": total,
            "brightness_thresh": 90.0,
            "variance_thresh":   150.0,
            "edge_thresh":       12.0,
            "accuracy": None,
            "updated_at": datetime.utcnow().isoformat(),
            "status": "basic"
        }

    # Midpoint thresholds between class means
    b_thresh = (mean([r["brightness"] for r in alive]) + mean([r["brightness"] for r in dead])) / 2
    v_thresh = (mean([r["variance"]   for r in alive]) + mean([r["variance"]   for r in dead])) / 2
    e_thresh = (mean([r["edge"]       for r in alive]) + mean([r["edge"]       for r in dead])) / 2

    # Estimate accuracy on training set
    correct = 0
    for r in rows:
        pred = "dead" if (r["brightness"] > b_thresh and r["variance"] < v_thresh) else "alive"
        if pred == r["label"]:
            correct += 1
    accuracy = correct / total if total > 0 else None

    return {
        "version": 2,
        "sample_count": total,
        "brightness_thresh": round(b_thresh, 4),
        "variance_thresh":   round(v_thresh, 4),
        "edge_thresh":       round(e_thresh, 4),
        "accuracy":          round(accuracy, 4) if accuracy else None,
        "updated_at":        datetime.utcnow().isoformat(),
        "status":            "trained"
    }

# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"service": "Charola Checker API", "status": "ok"}

@app.get("/health")
def health():
    with get_db() as conn:
        count = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    return {"status": "ok", "total_samples": count}

@app.post("/samples", status_code=201)
def upload_samples(batch: SampleBatch):
    """Upload correction samples from a device."""
    if not batch.samples:
        raise HTTPException(400, "Empty batch")

    valid_labels = {"alive", "dead"}
    for s in batch.samples:
        if s.label not in valid_labels:
            raise HTTPException(400, f"Invalid label: {s.label}")

    with get_db() as conn:
        conn.executemany(
            "INSERT INTO samples (brightness, variance, edge, label, device_id) VALUES (?,?,?,?,?)",
            [(s.brightness, s.variance, s.edge, s.label, s.device_id) for s in batch.samples]
        )
        # Invalidate cache
        conn.execute("DELETE FROM model_cache")
        conn.commit()

    return {"accepted": len(batch.samples)}

@app.get("/model", response_model=ModelResponse)
def get_model():
    """
    Returns the current best thresholds derived from all training samples.
    The iOS app calls this on launch to sync the latest model.
    """
    with get_db() as conn:
        # Try cache first
        cache = conn.execute("SELECT * FROM model_cache ORDER BY id DESC LIMIT 1").fetchone()
        if cache:
            return json.loads(cache["payload"])

        # Recompute
        model = compute_model(conn)
        conn.execute(
            "INSERT INTO model_cache (payload, sample_count) VALUES (?,?)",
            [json.dumps(model), model["sample_count"]]
        )
        conn.commit()
        return model

@app.get("/stats")
def stats():
    """Admin endpoint — summary stats."""
    with get_db() as conn:
        total     = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        alive_ct  = conn.execute("SELECT COUNT(*) FROM samples WHERE label='alive'").fetchone()[0]
        dead_ct   = conn.execute("SELECT COUNT(*) FROM samples WHERE label='dead'").fetchone()[0]
        devices   = conn.execute("SELECT COUNT(DISTINCT device_id) FROM samples WHERE device_id IS NOT NULL").fetchone()[0]
        per_day   = conn.execute("""
            SELECT date(created_at) as day, COUNT(*) as n
            FROM samples GROUP BY day ORDER BY day DESC LIMIT 14
        """).fetchall()

    return {
        "total_samples": total,
        "alive": alive_ct,
        "dead":  dead_ct,
        "devices": devices,
        "balance_ratio": round(alive_ct / dead_ct, 2) if dead_ct else None,
        "per_day": [{"day": r["day"], "count": r["n"]} for r in per_day]
    }

@app.delete("/samples/reset")
def reset_samples(secret: str):
    """Emergency reset — requires SECRET env var."""
    expected = os.environ.get("RESET_SECRET", "")
    if not expected or secret != expected:
        raise HTTPException(403, "Forbidden")
    with get_db() as conn:
        conn.execute("DELETE FROM samples")
        conn.execute("DELETE FROM model_cache")
        conn.commit()
    return {"deleted": True}
