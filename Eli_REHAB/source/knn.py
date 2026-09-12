import sys
from pathlib import Path

import pandas as pd

from sklearn.preprocessing import StandardScaler


sys.stdout.reconfigure(encoding="utf-8")


# ============================================================
# RUTAS
# ============================================================

BASE_PATH = Path(__file__).resolve().parent.parent
SPLITS_PATH = BASE_PATH / "data" / "splits"


# ============================================================
# 1. CARGAR DATOS
# ============================================================

train_df = pd.read_csv(SPLITS_PATH / "train.csv")
validation_df = pd.read_csv(SPLITS_PATH / "validation.csv")
test_df = pd.read_csv(SPLITS_PATH / "test.csv")

print("\n=== DATOS CARGADOS ===")
print(f"Train: {train_df.shape}")
print(f"Validation: {validation_df.shape}")
print(f"Test: {test_df.shape}")


# ============================================================
# 2. SEPARAR FEATURES Y ETIQUETA
# ============================================================

X_train = train_df.drop(columns=["activity"])
y_train = train_df["activity"]

X_validation = validation_df.drop(columns=["activity"])
y_validation = validation_df["activity"]

X_test = test_df.drop(columns=["activity"])
y_test = test_df["activity"]


# ============================================================
# 3. ESTANDARIZACIÓN
# ============================================================

scaler = StandardScaler()

# IMPORTANTE:
# El scaler aprende media y desviación SOLO de train.
X_train_scaled = scaler.fit_transform(X_train)

# Validation y test solamente se transforman.
X_validation_scaled = scaler.transform(X_validation)
X_test_scaled = scaler.transform(X_test)

print("\n=== PREPROCESAMIENTO ===")
print("StandardScaler ajustado únicamente con train.")
print(f"Features utilizadas: {X_train.shape[1]}")
print(f"Clases: {y_train.nunique()}")

# ============================================================
# 4. ENTRENAR DIFERENTES CONFIGURACIONES DE KNN
# ============================================================

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


RESULTS_PATH = BASE_PATH / "results"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)


# Configuraciones que vamos a comparar
k_values = [1, 3, 5, 7, 9]
weights_options = ["uniform", "distance"]
metrics = ["euclidean", "manhattan"]

resultados = []

print("\n=== ENTRENAMIENTO KNN ===")

for k in k_values:
    for weights in weights_options:
        for metric in metrics:

            modelo = KNeighborsClassifier(
                n_neighbors=k,
                weights=weights,
                metric=metric
            )

            # Entrenamiento SOLO con train
            modelo.fit(X_train_scaled, y_train)

            # Evaluación para selección SOLO con validation
            pred_validation = modelo.predict(X_validation_scaled)

            accuracy = accuracy_score(
                y_validation,
                pred_validation
            )

            precision = precision_score(
                y_validation,
                pred_validation,
                average="macro",
                zero_division=0
            )

            recall = recall_score(
                y_validation,
                pred_validation,
                average="macro",
                zero_division=0
            )

            f1 = f1_score(
                y_validation,
                pred_validation,
                average="macro",
                zero_division=0
            )

            resultados.append({
                "k": k,
                "weights": weights,
                "metric": metric,
                "accuracy_validation": accuracy,
                "precision_macro_validation": precision,
                "recall_macro_validation": recall,
                "f1_macro_validation": f1
            })

            print(
                f"k={k:<2} | "
                f"weights={weights:<8} | "
                f"metric={metric:<9} | "
                f"Accuracy={accuracy:.4f} | "
                f"F1 macro={f1:.4f}"
            )


# ============================================================
# 5. COMPARAR CONFIGURACIONES
# ============================================================

resultados_df = pd.DataFrame(resultados)

resultados_df = resultados_df.sort_values(
    by="f1_macro_validation",
    ascending=False
).reset_index(drop=True)

resultados_df.to_csv(
    RESULTS_PATH / "knn_results.csv",
    index=False
)

print("\n=== MEJORES CONFIGURACIONES KNN ===")

print(
    resultados_df.head(10).to_string(index=False)
)


mejor = resultados_df.iloc[0]

print("\n=== MEJOR KNN SEGÚN VALIDATION ===")

print(f"k: {int(mejor['k'])}")
print(f"weights: {mejor['weights']}")
print(f"metric: {mejor['metric']}")
print(
    f"Accuracy validation: "
    f"{mejor['accuracy_validation']:.4f}"
)
print(
    f"F1 macro validation: "
    f"{mejor['f1_macro_validation']:.4f}"
)

print(
    "\nResultados guardados en "
    "results/knn_results.csv"
)