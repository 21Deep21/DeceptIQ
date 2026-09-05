"""Group-aware stacking ensemble (Phase 2).

Why custom instead of sklearn's StackingClassifier:
  1. sklearn's fit(X, y) takes no `groups` argument, so its internal
     cross_val_predict cannot directly enforce domain-grouped folds.
  2. Our base learners are full pipelines consuming RAW URL strings;
     an explicit implementation keeps the data flow auditable.

Meta-learner (Logistic Regression) is trained on OUT-OF-FOLD predicted
probabilities produced by StratifiedGroupKFold over the training split:
it never trains on in-sample base predictions, which would be overfit
and would mislead the meta weights. Groups (registrable domains) never
cross folds.
"""

from __future__ import annotations

import logging
from typing import Any, List, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

logger = logging.getLogger(__name__)


class GroupedStackingClassifier:
    """fit(X, y, groups) where X is an array-like of raw URL strings."""

    def __init__(self, base_estimators: List[Tuple[str, Any]], meta_estimator: Any,
                 n_folds: int = 5, seed: int = 42):
        self.base_estimators = list(base_estimators)  # [(name, UNFITTED pipeline)]
        self.meta_estimator = meta_estimator
        self.n_folds = n_folds
        self.seed = seed

    def fit(self, X, y, groups):
        skf = StratifiedGroupKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.seed
        )
        oof = np.zeros((len(X), len(self.base_estimators)), dtype=np.float64)
        for fold, (tr, te) in enumerate(skf.split(X, y, groups), start=1):
            for j, (name, proto) in enumerate(self.base_estimators):
                pipe = clone(proto)
                pipe.fit(X[tr], y[tr])
                oof[te, j] = pipe.predict_proba(X[te])[:, 1]
            logger.info("stacking: out-of-fold pass %d/%d complete", fold, self.n_folds)

        self.meta_ = clone(self.meta_estimator).fit(oof, y)
        names = [name for name, _ in self.base_estimators]
        coefs = np.round(np.asarray(self.meta_.coef_).ravel(), 4)
        logger.info("stacking meta-learner coefficients: %s", dict(zip(names, coefs.tolist())))

        self.fitted_bases_ = [
            (name, clone(proto).fit(X, y)) for name, proto in self.base_estimators
        ]
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X) -> np.ndarray:
        base = np.column_stack(
            [pipe.predict_proba(X)[:, 1] for _, pipe in self.fitted_bases_]
        )
        p1 = np.asarray(self.meta_.predict_proba(base))[:, 1]
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
