"""
Phase 9: Machine Learning grid search.

Objetivos:
1. AF continuo  -> regresión
2. AF categórico -> clasificación

Se evalúan múltiples hiperparámetros mediante validación cruzada.
"""

import logging
import pandas as pd

from sklearn.model_selection import GridSearchCV, cross_val_score

# =========================================================
# REGRESIÓN
# =========================================================

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    LogisticRegression,
)

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    ExtraTreesRegressor,
    ExtraTreesClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)

from sklearn.svm import (
    SVR,
    SVC,
)

from sklearn.neighbors import (
    KNeighborsRegressor,
    KNeighborsClassifier,
)

from src.config import (
    SEED,
    CV_FOLDS,
)

log = logging.getLogger(__name__)


# =========================================================
# REGRESIÓN
# =========================================================

def regression_grid_search(X, y):
    """
    Grid search para AF continuo.
    """

    log.info("Running regression grid search...")

    results = []

    # ---------------------------------------------------------
    # Linear Regression
    # ---------------------------------------------------------

    scores = cross_val_score(
        LinearRegression(),
        X,
        y,
        cv=CV_FOLDS,
        scoring="r2",
        n_jobs=-1,
    )

    results.append({
        "modelo": "LinearRegression",
        "best_score": scores.mean(),
        "best_params": "{}",
    })

    # ---------------------------------------------------------
    # Modelos con Grid Search
    # ---------------------------------------------------------

    models = [

        (
            "Ridge",
            Ridge(),
            {
                "alpha": [0.01, 0.1, 1, 10, 100],
            },
        ),

        (
            "Lasso",
            Lasso(max_iter=10000),
            {
                "alpha": [0.0001, 0.001, 0.01, 0.1, 1],
            },
        ),

        (
            "RandomForestRegressor",
            RandomForestRegressor(
                random_state=SEED,
            ),
            {
                "n_estimators": [100, 300],
                "max_depth": [None, 10, 20],
            },
        ),

        (
            "ExtraTreesRegressor",
            ExtraTreesRegressor(
                random_state=SEED,
            ),
            {
                "n_estimators": [100, 300],
                "max_depth": [None, 10, 20],
            },
        ),

        (
            "GradientBoostingRegressor",
            GradientBoostingRegressor(
                random_state=SEED,
            ),
            {
                "n_estimators": [100, 300],
                "learning_rate": [0.01, 0.05, 0.1],
            },
        ),

        (
            "SVR",
            SVR(),
            {
                "C": [0.1, 1, 10],
                "gamma": ["scale", "auto"],
                "kernel": ["rbf"],
            },
        ),

        (
            "KNeighborsRegressor",
            KNeighborsRegressor(),
            {
                "n_neighbors": [3, 5, 7, 11, 15],
                "weights": ["uniform", "distance"],
            },
        ),
    ]

    for name, model, grid in models:

        search = GridSearchCV(
            model,
            grid,
            cv=CV_FOLDS,
            scoring="r2",
            n_jobs=-1,
        )

        search.fit(X, y)

        results.append({
            "modelo": name,
            "best_score": search.best_score_,
            "best_params": str(search.best_params_),
        })

        log.info(
            "%s -> %.4f",
            name,
            search.best_score_,
        )

    # Crear DataFrame con los resultados
    df_results = pd.DataFrame(results)

    # Ordenar de mejor a peor
    df_results = df_results.sort_values(
        by="best_score",
        ascending=False
    ).reset_index(drop=True)

    return df_results


# =========================================================
# CLASIFICACIÓN
# =========================================================

def classification_grid_search(X, y):
    """
    Grid search para AF categórico.
    """

    log.info("Running classification grid search...")

    results = []

    models = [

        (
            "LogisticRegression",
            LogisticRegression(
                max_iter=5000,
                random_state=SEED,
            ),
            {
                "C": [0.01, 0.1, 1, 10, 100],
            },
        ),

        (
            "RandomForestClassifier",
            RandomForestClassifier(
                random_state=SEED,
            ),
            {
                "n_estimators": [100, 300],
                "max_depth": [None, 10, 20],
            },
        ),

        (
            "ExtraTreesClassifier",
            ExtraTreesClassifier(
                random_state=SEED,
            ),
            {
                "n_estimators": [100, 300],
                "max_depth": [None, 10, 20],
            },
        ),

        (
            "GradientBoostingClassifier",
            GradientBoostingClassifier(
                random_state=SEED,
            ),
            {
                "n_estimators": [100, 300],
                "learning_rate": [0.01, 0.05, 0.1],
            },
        ),

        (
            "SVC",
            SVC(),
            {
                "C": [0.1, 1, 10],
                "gamma": ["scale", "auto"],
                "kernel": ["rbf"],
            },
        ),

        (
            "KNeighborsClassifier",
            KNeighborsClassifier(),
            {
                "n_neighbors": [3, 5, 7, 11, 15],
                "weights": ["uniform", "distance"],
            },
        ),
    ]

    for name, model, grid in models:

        search = GridSearchCV(
        model,
        grid,
        cv=CV_FOLDS,
        scoring="accuracy",
        n_jobs=-1,
        )

        search.fit(X, y)

        results.append({
            "modelo": name,
            "best_score": search.best_score_,
            "best_params": str(search.best_params_),
        })

        log.info(
            "%s -> %.4f",
            name,
            search.best_score_,
        )

    # Crear DataFrame con los resultados
    df_results = pd.DataFrame(results)

    # Ordenar de mejor a peor
    df_results = df_results.sort_values(
        by="best_score",
        ascending=False
    ).reset_index(drop=True)

    return df_results