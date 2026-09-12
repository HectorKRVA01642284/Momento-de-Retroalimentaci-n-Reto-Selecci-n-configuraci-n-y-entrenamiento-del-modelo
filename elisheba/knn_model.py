import sys
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import ConfusionMatrixDisplay


BASE_DIR = Path(__file__).resolve().parent.parent

# Reutilizar las mismas comprobaciones y métricas de los otros modelos.
sys.path.insert(0, str(BASE_DIR / "hector"))

from decision_tree_model import (
    CLASSES,
    load_data,
    calculate_metrics,
)


RESULTS_DIR = BASE_DIR / "results" / "knn"


def main():
    X_train, y_train = load_data("train.csv")
    X_validation, y_validation = load_data("validation.csv")

    print(f"Entrenamiento: {len(X_train)} muestras")
    print(f"Validación: {len(X_validation)} muestras")

    k_values = [1, 3, 5, 7, 9]
    weights_options = ["uniform", "distance"]
    distance_metrics = ["euclidean", "manhattan"]

    rows = []
    best_params = None
    best_score = -1.0
    best_predictions = None

    configuration_id = 0

    for k in k_values:
        for weights in weights_options:
            for metric in distance_metrics:
                configuration_id += 1

                params = {
                    "n_neighbors": k,
                    "weights": weights,
                    "metric": metric,
                }

                # El escalador se ajusta solo durante fit con train.
                # predict aplica esa transformación a validation.
                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("knn", KNeighborsClassifier(**params)),
                ])

                model.fit(X_train, y_train)

                predictions = model.predict(X_validation)
                metrics = calculate_metrics(
                    y_validation, predictions
                )

                rows.append({
                    "configuration_id": configuration_id,
                    "model": "KNN",
                    "split": "validation",
                    **params,
                    **metrics,
                    "train_f1_macro": calculate_metrics(
                        y_train, model.predict(X_train)
                    )["f1_macro"],
                })

                print(
                    f"{configuration_id:02d} | {params} "
                    f"| F1 macro: {metrics['f1_macro']:.4f}"
                )

                # En empate exacto, conservar la primera probada.
                if metrics["f1_macro"] > best_score:
                    best_score = metrics["f1_macro"]
                    best_params = params.copy()
                    best_predictions = predictions.copy()

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
        "model": "KNN",
        "selection_metric": "validation_f1_macro",
        "preprocessing": "StandardScaler ajustado solo con train",
        "best_params": best_params,
        "validation_metrics": {
            key: float(value)
            for key, value in calculate_metrics(
                y_validation, best_predictions
            ).items()
        },
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

    ax.set_title("KNN — Validación")
    ax.set_xlabel("Actividad predicha")
    ax.set_ylabel("Actividad real")
    fig.tight_layout()

    fig.savefig(
        RESULTS_DIR / "validation_confusion_matrix.png",
        dpi=180,
    )
    plt.close(fig)

    print("\nMejor configuración:", best_params)
    print(f"F1 macro de validación: {best_score:.4f}")
    print(f"Resultados guardados en: {RESULTS_DIR}")


if __name__ == "__main__":
    main()