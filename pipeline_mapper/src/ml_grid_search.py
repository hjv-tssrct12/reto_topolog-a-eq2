"""
Phase 9: Machine Learning grid search.

Objetivos:
1. AF continuo   → regresión   (métrica: R²)
2. AF categórico → clasificación (métrica: f1_macro + reporte por clase)
"""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR

from src.config import CV_FOLDS, SEED

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regresión
# ---------------------------------------------------------------------------

def regression_grid_search(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Grid search para AF continuo.  Métrica: R²."""

    log.info("Running regression grid search...")
    results = []

    # Linear Regression (sin hiperparámetros)
    scores = cross_val_score(
        LinearRegression(), X, y, cv=CV_FOLDS, scoring="r2", n_jobs=-1,
    )
    results.append({
        "modelo"     : "LinearRegression",
        "best_score" : round(float(scores.mean()), 4),
        "best_params": "{}",
    })

    models = [
        (
            "Ridge", Ridge(),
            {"alpha": [0.01, 0.1, 1, 10, 100]},
        ),
        (
            "Lasso", Lasso(max_iter=10000),
            {"alpha": [0.0001, 0.001, 0.01, 0.1, 1]},
        ),
        (
            "RandomForestRegressor",
            RandomForestRegressor(random_state=SEED),
            {"n_estimators": [100, 300], "max_depth": [None, 10, 20]},
        ),
        (
            "ExtraTreesRegressor",
            ExtraTreesRegressor(random_state=SEED),
            {"n_estimators": [100, 300], "max_depth": [None, 10, 20]},
        ),
        (
            "GradientBoostingRegressor",
            GradientBoostingRegressor(random_state=SEED),
            {"n_estimators": [100, 300], "learning_rate": [0.01, 0.05, 0.1]},
        ),
        (
            "SVR", SVR(),
            {"C": [0.1, 1, 10], "gamma": ["scale", "auto"], "kernel": ["rbf"]},
        ),
        (
            "KNeighborsRegressor", KNeighborsRegressor(),
            {"n_neighbors": [3, 5, 7, 11, 15], "weights": ["uniform", "distance"]},
        ),
    ]

    for name, model, grid in models:
        search = GridSearchCV(
            model, grid, cv=CV_FOLDS, scoring="r2", n_jobs=-1,
        )
        search.fit(X, y)
        results.append({
            "modelo"     : name,
            "best_score" : round(float(search.best_score_), 4),
            "best_params": str(search.best_params_),
        })
        log.info("%s -> %.4f", name, search.best_score_)

    df = pd.DataFrame(results).sort_values("best_score", ascending=False)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Clasificación
# ---------------------------------------------------------------------------

def classification_grid_search(
    X: np.ndarray,
    y: np.ndarray,
) -> pd.DataFrame:
    """Grid search para AF categórico.

    Cambios respecto a la versión anterior
    ---------------------------------------
    - Métrica principal: **f1_macro** (en lugar de accuracy).
      f1_macro promedia el F1 de cada clase sin ponderar por frecuencia,
      lo que es más informativo con clases aproximadamente balanceadas.
    - Se agrega un **reporte por clase** (precisión, recall, F1 para
      Bajo / Medio / Alto) usando el mejor modelo encontrado.
    - Se usa StratifiedKFold para garantizar que cada fold mantenga la
      proporción de clases.

    Columnas del DataFrame devuelto
    --------------------------------
    modelo, f1_macro, accuracy_cv, best_params,
    precision_Bajo, recall_Bajo, f1_Bajo,
    precision_Medio, recall_Medio, f1_Medio,
    precision_Alto, recall_Alto, f1_Alto,
    support_Bajo, support_Medio, support_Alto
    """

    log.info("Running classification grid search (scoring=f1_macro)...")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    models = [
        (
            "LogisticRegression",
            LogisticRegression(max_iter=5000, random_state=SEED),
            {"C": [0.01, 0.1, 1, 10, 100]},
        ),
        (
            "RandomForestClassifier",
            RandomForestClassifier(random_state=SEED),
            {"n_estimators": [100, 300], "max_depth": [None, 10, 20]},
        ),
        (
            "ExtraTreesClassifier",
            ExtraTreesClassifier(random_state=SEED),
            {"n_estimators": [100, 300], "max_depth": [None, 10, 20]},
        ),
        (
            "GradientBoostingClassifier",
            GradientBoostingClassifier(random_state=SEED),
            {"n_estimators": [100, 300], "learning_rate": [0.01, 0.05, 0.1]},
        ),
        (
            "SVC",
            SVC(random_state=SEED),
            {"C": [0.1, 1, 10], "gamma": ["scale", "auto"], "kernel": ["rbf"]},
        ),
        (
            "KNeighborsClassifier",
            KNeighborsClassifier(),
            {"n_neighbors": [3, 5, 7, 11, 15], "weights": ["uniform", "distance"]},
        ),
    ]

    results = []

    for name, model, grid in models:

        # ── Grid search con f1_macro ───────────────────────────────────────
        search = GridSearchCV(
            model, grid, cv=cv, scoring="f1_macro", n_jobs=-1, refit=True,
        )
        search.fit(X, y)

        # ── Accuracy CV con los mejores parámetros (referencia) ───────────
        acc_scores = cross_val_score(
            search.best_estimator_, X, y, cv=cv, scoring="accuracy", n_jobs=-1,
        )

        # ── Reporte por clase con el mejor estimador (CV manual) ──────────
        # Acumulamos predicciones de todos los folds para el reporte
        best_est = search.best_estimator_
        y_true_all, y_pred_all = [], []

        for train_idx, val_idx in cv.split(X, y):
            best_est.fit(X[train_idx], y[train_idx])
            y_true_all.extend(y[val_idx])
            y_pred_all.extend(best_est.predict(X[val_idx]))

        report = classification_report(
            y_true_all, y_pred_all,
            output_dict=True,
            zero_division=0,
        )

        row = {
            "modelo"     : name,
            "f1_macro"   : round(float(search.best_score_), 4),
            "accuracy_cv": round(float(acc_scores.mean()),  4),
            "best_params": str(search.best_params_),
        }

        # Métricas por clase
        for clase in ["Bajo", "Medio", "Alto"]:
            if clase in report:
                row[f"precision_{clase}"] = round(report[clase]["precision"], 4)
                row[f"recall_{clase}"]    = round(report[clase]["recall"],    4)
                row[f"f1_{clase}"]        = round(report[clase]["f1-score"],  4)
                row[f"support_{clase}"]   = int(report[clase]["support"])
            else:
                row[f"precision_{clase}"] = None
                row[f"recall_{clase}"]    = None
                row[f"f1_{clase}"]        = None
                row[f"support_{clase}"]   = 0

        results.append(row)
        log.info(
            "%s → f1_macro=%.4f  acc=%.4f  "
            "[Bajo f1=%.3f | Medio f1=%.3f | Alto f1=%.3f]",
            name,
            row["f1_macro"], row["accuracy_cv"],
            row.get("f1_Bajo",  0),
            row.get("f1_Medio", 0),
            row.get("f1_Alto",  0),
        )

    df = pd.DataFrame(results).sort_values("f1_macro", ascending=False)
    return df.reset_index(drop=True)