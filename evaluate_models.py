import sys
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay


BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "hector"))

from decision_tree_model import (
    CLASSES,
    load_data,
    calculate_metrics,
)


OUTPUT_DIR = BASE_DIR / "results" / "final_test"


def read_params(folder):
    path = BASE_DIR / "results" / folder / "best_configuration.json"
    with open(path, encoding="utf-8") as file:
        return json.load(file)["best_params"]


def main():
    X_train, y_train = load_data("train.csv")
    X_test, y_test = load_data("test.csv")

    # Recuperar las configuraciones elegidas en validación.
    models = {
        "decision_tree": DecisionTreeClassifier(
            **read_params("decision_tree"),
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            **read_params("random_forest"),
            random_state=42,
            n_jobs=-1,
        ),
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(**read_params("knn"))),
        ]),
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(**read_params("svm"))),
        ]),
    }

    names = {
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest",
        "knn": "KNN",
        "svm": "SVM",
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    print(f"Entrenamiento: {len(X_train)} muestras")
    print(f"Prueba: {len(X_test)} muestras")
    print("Modelo seleccionado por validación: Random Forest")

    for key, model in models.items():
        name = names[key]
        print(f"\nEvaluando {name}...", flush=True)

        # No se ajustan parámetros ni escaladores con test.
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = calculate_metrics(y_test, predictions)

        rows.append({
            "model": name,
            "split": "test",
            "selected_on_validation": key == "random_forest",
            **metrics,
        })

        pd.DataFrame({
            "test_row": range(len(y_test)),
            "activity_real": y_test.to_numpy(),
            "activity_predicted": predictions,
        }).to_csv(
            OUTPUT_DIR / f"{key}_predictions.csv",
            index=False,
        )

        fig, ax = plt.subplots(figsize=(10, 9))
        display = ConfusionMatrixDisplay.from_predictions(
            y_test,
            predictions,
            labels=CLASSES,
            display_labels=CLASSES,
            cmap="Blues",
            values_format="d",
            colorbar=False,
            ax=ax,
        )

        pd.DataFrame(
            display.confusion_matrix,
            index=pd.Index(CLASSES, name="activity_real"),
            columns=CLASSES,
        ).to_csv(OUTPUT_DIR / f"{key}_confusion_matrix.csv")

        ax.set_title(f"{name} — Test")
        ax.set_xlabel("Actividad predicha")
        ax.set_ylabel("Actividad real")
        fig.tight_layout()
        fig.savefig(
            OUTPUT_DIR / f"{key}_confusion_matrix.png",
            dpi=180,
        )
        plt.close(fig)

        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 macro: {metrics['f1_macro']:.4f}")

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "test_metrics.csv", index=False)

    print("\n=== RESULTADOS DE TEST ===")
    print(results.to_string(index=False))
    print(f"\nArchivos guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()