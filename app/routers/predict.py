from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse

from .. import models, schemas
from ..core.config import settings
from ..deps import require_role

router = APIRouter(prefix="/predict", tags=["predict"])


@lru_cache(maxsize=1)
def _load_model():
    import joblib

    model_path = Path(__file__).resolve().parents[3] / settings.model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)


@lru_cache(maxsize=1)
def _load_schema() -> Dict[str, Any]:
    schema_path = Path(__file__).resolve().parents[3] / settings.schema_path
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found at {schema_path}")
    return json.loads(schema_path.read_text(encoding="utf-8"))


@router.get("/schema")
def get_schema(_user: models.User = Depends(require_role("admin", "clinician"))):
    try:
        return _load_schema()
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/", response_model=schemas.PredictResponse)
def predict(
    payload: schemas.PredictRequest,
    _user: models.User = Depends(require_role("admin", "clinician")),
):
    try:
        model = _load_model()
        schema = _load_schema()
        feature_columns = schema.get("feature_columns") or []

        if not feature_columns:
            raise ValueError("Missing feature_columns in schema")

        row = {c: payload.features.get(c) for c in feature_columns}
        X = pd.DataFrame([row], columns=feature_columns)

        proba_arr = model.predict_proba(X)[0]
        proba = float(proba_arr[1])
        pred = int(proba >= 0.5)

        label = "ASD Risk Detected" if pred == 1 else "Low Risk"
        confidence = proba if pred == 1 else 1.0 - proba

        return schemas.PredictResponse(
            label=label,
            prediction=pred,
            proba_asd=proba,
            confidence=float(confidence),
            probabilities={"Low Risk": float(proba_arr[0]), "ASD Risk": float(proba_arr[1])},
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Prediction failed: {e}")


@router.get("/metrics")
def get_metrics(_user: models.User = Depends(require_role("admin", "clinician"))):
    metrics_path = Path(__file__).resolve().parents[3] / "ml" / "artifacts_tabular" / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Metrics file not found")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


@router.get("/confusion-matrix")
def get_confusion_matrix(_user: models.User = Depends(require_role("admin", "clinician"))):
    img_path = Path(__file__).resolve().parents[3] / "ml" / "artifacts_tabular" / "confusion_matrix.png"
    if not img_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Confusion matrix image not found")
    return FileResponse(str(img_path), media_type="image/png")
