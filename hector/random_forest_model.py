import json

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import ConfusionMatrixDisplay

from decision_tree_model import (
    BASE_DIR,
    CLASSES,
    FEATURES,
    load_data,
    calculate_metrics,
)


RESULTS_DIR = BASE_DIR / "results" / "random_forest"


def main():
    X_train, y_train = load_data("train.csv")
    X_validation, y_validation = load_data("validation.csv")

    print(f"Entrenamiento: {len(X_train)} muestras")
    print(f"Validación: {len(X_validation)} muestras")

    configurations = ParameterGrid({
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
        "min_samples_leaf": [1, 5],
    })

    rows = []
    best_model = None
    best_params = None
    best_score = -1.0
    best_predictions = None

    for configuration_id, params in enumerate(configurations, start=1):
        model = RandomForestClassifier(
            **params,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_validation)
        metrics = calculate_metrics(y_validation, predictions)

        rows.append({
            "configuration_id": configuration_id,
            "model": "Random Forest",
            "split": "validation",
            "n_estimators": params["n_estimators"],
            "max_depth": (
                params["max_depth"]
                if params["max_depth"] is not None
                else "None"
            ),
            "min_samples_leaf": params["min_samples_leaf"],
            "max_features": "sqrt",
            **metrics,
            "train_f1_macro": calculate_metrics(
                y_train, model.predict(X_train)
            )["f1_macro"],
        })

        print(
            f"{configuration_id:02d} | {params} "
            f"| F1 macro: {metrics['f1_macro']:.4f}"
        )

        # En empate exacto, conservar la primera configuración probada.
        if metrics["f1_macro"] > best_score:
            best_score = metrics["f1_macro"]
            best_model = model
            best_params = params
            best_predictions = predictions

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(rows).sort_values(
        "f1_macro",
        ascending=False,
        kind="stable",
    )
    results.to_csv(
        RESULTS_DIR / "validation_metrics.csv",
        index=False,
    )

    summary = {
        "model": "Random Forest",
        "selection_metric": "validation_f1_macro",
        "random_state": 42,
        "best_params": {
            **best_params,
            "max_features": "sqrt",
        },
        "validation_metrics": calculate_metrics(
            y_validation, best_predictions
        ),
        "train_samples": len(X_train),
        "validation_samples": len(X_validation),
    }

    with open(
        RESULTS_DIR / "best_configuration.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    fig, ax = plt.subplots(figsize=(10, 9))
    ConfusionMatrixDisplay.from_predictions(
        y_validation,
        best_predictions,
        labels=CLASSES,
        display_labels=CLASSES,
        cmap="Blues",
        values_format="d",
        colorbar=False,
        ax=ax,
    )
    ax.set_title("Random Forest — Validación")
    ax.set_xlabel("Actividad predicha")
    ax.set_ylabel("Actividad real")
    fig.tight_layout()
    fig.savefig(
        RESULTS_DIR / "validation_confusion_matrix.png",
        dpi=180,
    )
    plt.close(fig)

    # Importancia basada en reducción de impureza dentro del bosque.
    # Describe el modelo; no demuestra causalidad.
    importances = pd.DataFrame({
        "feature": FEATURES,
        "importance": best_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importances.to_csv(
        RESULTS_DIR / "feature_importances.csv",
        index=False,
    )

    print("\nMejor configuración:", best_params)
    print(f"F1 macro de validación: {best_score:.4f}")
    print(f"Resultados guardados en: {RESULTS_DIR}")


if __name__ == "__main__":
    main()