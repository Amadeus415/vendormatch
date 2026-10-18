"""Stage 3: the learned pair scorer. Gradient-boosted trees over the features in features.py.

Trained on the dev split only (bench/train.py). The shipped model file is small and is loaded by the
pipeline and CLI.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES

MODEL_PATH = Path(__file__).with_name("model.pkl")


def _lr_matrix(X: np.ndarray) -> np.ndarray:
    """Logistic regression cannot take NaN: fill with 0 and add a was-missing column per feature."""
    missing = np.isnan(X)
    return np.hstack([np.nan_to_num(X, nan=0.0), missing.astype(np.float32)])


def make_model(kind: str, seed: int = 0):
    if kind == "gbm":
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, random_state=seed
        )
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    raise ValueError(kind)


@dataclass
class PairScorer:
    kind: str = "gbm"
    threshold: float = 0.5
    band: tuple[float, float] = (0.2, 0.8)  # scores in this range go to the LLM when one is enabled
    model: object = None
    meta: dict = field(default_factory=dict)

    def fit(self, X: np.ndarray, y: np.ndarray, seed: int = 0) -> PairScorer:
        self.model = make_model(self.kind, seed)
        self.model.fit(_lr_matrix(X) if self.kind == "logreg" else X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.zeros(0)
        Xm = _lr_matrix(X) if self.kind == "logreg" else X
        return self.model.predict_proba(Xm)[:, 1]

    def save(self, path: Path = MODEL_PATH) -> None:
        with path.open("wb") as f:
            pickle.dump({"kind": self.kind, "threshold": self.threshold, "band": self.band,
                         "features": FEATURES, "model": self.model, "meta": self.meta}, f)  # fmt: skip
        path.with_suffix(".json").write_text(
            json.dumps({"kind": self.kind, "threshold": self.threshold, "band": self.band,
                        "features": FEATURES, "meta": self.meta}, indent=2) + "\n"  # fmt: skip
        )

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> PairScorer:
        with path.open("rb") as f:
            d = pickle.load(f)  # noqa: S301 - our own file, shipped with the package
        if d["features"] != FEATURES:
            raise RuntimeError("model was trained on a different feature set; run `make train`")
        return cls(kind=d["kind"], threshold=d["threshold"], band=tuple(d["band"]), model=d["model"],
                   meta=d.get("meta", {}))  # fmt: skip
