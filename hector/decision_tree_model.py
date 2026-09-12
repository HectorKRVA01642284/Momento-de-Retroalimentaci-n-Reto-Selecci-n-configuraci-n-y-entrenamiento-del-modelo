from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    ConfusionMatrixDisplay,
)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results" / "decision_tree"

FEATURES = [
    f"{channel}_{stat}"
    for channel in ["f1", "f2", "f3", "f4", "f5", "pitch3"]
    for stat in ["mean", "std", "min", "max"]
]
CLASSES = [f"{i:03d}" for i in range(16) if i != 14]


def load_data(filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Falta {path.name} en data/. "
            "Espera las particiones definitivas de tu compañera."
        )

    df = pd.read_csv(path, dtype={"activity": str})

    if set(df.columns) != set(FEATURES + ["activity"]):
        raise ValueError(
            f"{filename}: se esperaban las 24 características y activity."
        )

    if df.empty or df.isna().any().any():
        raise ValueError(f"{filename}: está vacío o contiene valores faltantes.")

    X = df[FEATURES].apply(pd.to_numeric, errors="raise")
    y = df["activity"].str.strip().str.zfill(3)

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError(f"{filename}: contiene valores infinitos.")

    if set(y) != set(CLASSES):
        raise ValueError(
            f"{filename}: deben estar presentes las 15 clases acordadas."
        )

    return X, y


def calculate_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASSES,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }


def main():
    X_train, y_train = load_data("train.csv")
    X_validation, y_validation = load_data("validation.csv")

    print(f"Entrenamiento: {len(X_train)} muestras")
    print(f"Validación: {len(X_validation)} muestras")

    # Los árboles trabajan directamente con las variables sin estandarizar.
    configurations = ParameterGrid({
        "max_depth": [3, 5, 10, None],
        "min_samples_leaf": [1, 2, 5, 10],
    })

    rows = []
    best_model = None
    best_params = None
    best_score = -1.0
    best_predictions = None

    for configuration_id, params in enumerate(configurations, start=1):
        model = DecisionTreeClassifier(
            **params,
            random_state=42,
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_validation)
        metrics = calculate_metrics(y_validation, predictions)

        rows.append({
            "configuration_id": configuration_id,
            "model": "Decision Tree",
            "split": "validation",
            "max_depth": (
                params["max_depth"]
                if params["max_depth"] is not None
                else "None"
            ),
            "min_samples_leaf": params["min_samples_leaf"],
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
        "model": "Decision Tree",
        "selection_metric": "validation_f1_macro",
        "random_state": 42,
        "best_params": best_params,
        "validation_metrics": calculate_metrics(
            y_validation, best_predictions
        ),
        "train_samples": len(X_train),
        "validation_samples": len(X_validation),
        "tree_depth": int(best_model.get_depth()),
        "tree_leaves": int(best_model.get_n_leaves()),
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
    ax.set_title("Decision Tree — Validación")
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