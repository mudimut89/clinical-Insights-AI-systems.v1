from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from ucimlrepo import fetch_ucirepo
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "ucimlrepo is required. Install dependencies from requirements.txt"
    ) from e


def _sanitize_col(c: str) -> str:
    c = c.strip()
    c = re.sub(r"\s+", "_", c)
    c = re.sub(r"[^a-zA-Z0-9_]+", "", c)
    return c.lower()


def load_uci_children() -> Tuple[pd.DataFrame, pd.Series]:
    # UCI id=419: Autistic Spectrum Disorder Screening Data for Children
    ds = fetch_ucirepo(id=419)

    X = ds.data.features.copy()
    y = ds.data.targets.copy()

    # Normalize column names
    X.columns = [_sanitize_col(c) for c in X.columns]
    if isinstance(y, pd.DataFrame):
        # Target column name is usually 'class'
        if y.shape[1] == 1:
            y = y.iloc[:, 0]
        else:
            # take the first target if multiple
            y = y.iloc[:, 0]

    y = y.astype(str).str.strip().str.lower()

    # Common labels in this family of datasets include 'yes'/'no'
    # Convert to 1/0
    y_bin = y.map({"yes": 1, "no": 0})
    if y_bin.isna().any():
        # If labels are different, fallback to factorization
        y_bin = pd.Series(pd.factorize(y)[0], index=y.index)

    return X, y_bin


def save_confusion_matrix(cm: np.ndarray, labels: list[str], out_path: Path) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()

    project_root = Path(args.project_root)
    artifacts_dir = project_root / "ml" / "artifacts_tabular"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_uci_children()

    # Identify column types
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )

    base_model = LogisticRegression(max_iter=2000)

    clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", CalibratedClassifierCV(base_model, method="isotonic", cv=3)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    save_confusion_matrix(cm, ["no_asd", "asd"], artifacts_dir / "confusion_matrix.png")

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["no_asd", "asd"], output_dict=True
        ),
    }

    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Confidence samples (probability output)
    conf_samples = []
    for i, (p, pred, yt) in enumerate(zip(y_proba[:50], y_pred[:50], y_test.iloc[:50])):
        conf_samples.append(
            {
                "true": int(yt),
                "pred": int(pred),
                "confidence": float(p if pred == 1 else 1 - p),
                "proba_asd": float(p),
            }
        )
    (artifacts_dir / "confidence_samples.json").write_text(
        json.dumps(conf_samples, indent=2), encoding="utf-8"
    )

    # Persist model using joblib
    import joblib

    joblib.dump(clf, artifacts_dir / "model.joblib")

    (artifacts_dir / "schema.json").write_text(
        json.dumps({"feature_columns": list(X.columns)}, indent=2), encoding="utf-8"
    )

    print(f"Saved artifacts to: {artifacts_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
