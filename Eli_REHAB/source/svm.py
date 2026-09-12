import sys
from pathlib import Path

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


sys.stdout.reconfigure(encoding="utf-8")


# ============================================================
# RUTAS
# ============================================================

BASE_PATH = Path(__file__).resolve().parent.parent

SPLITS_PATH = BASE_PATH / "data" / "splits"
RESULTS_PATH = BASE_PATH / "results"

RESULTS_PATH.mkdir(parents=True, exist_ok=True)


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

# El scaler aprende SOLO de train
X_train_scaled = scaler.fit_transform(X_train)

# Validation y test solo se transforman
X_validation_scaled = scaler.transform(X_validation)
X_test_scaled = scaler.transform(X_test)

print("\n=== PREPROCESAMIENTO ===")
print("StandardScaler ajustado únicamente con train.")
print(f"Features utilizadas: {X_train.shape[1]}")
print(f"Clases: {y_train.nunique()}")


# ============================================================
# 4. CONFIGURACIONES DE SVM
# ============================================================

configuraciones = []

# Kernel lineal
for C in [0.1, 1, 10]:
    configuraciones.append({
        "kernel": "linear",
        "C": C,
        "gamma": "N/A"
    })

# Kernel RBF
for C in [0.1, 1, 10]:
    for gamma in ["scale", "auto"]:
        configuraciones.append({
            "kernel": "rbf",
            "C": C,
            "gamma": gamma
        })


# ============================================================
# 5. ENTRENAMIENTO Y VALIDACIÓN
# ============================================================

resultados = []

print("\n=== ENTRENAMIENTO SVM ===")

for config in configuraciones:

    kernel = config["kernel"]
    C = config["C"]
    gamma = config["gamma"]

    if kernel == "linear":

        modelo = SVC(
            kernel=kernel,
            C=C
        )

    else:

        modelo = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma
        )

    # Entrenamiento SOLO con train
    modelo.fit(X_train_scaled, y_train)

    # Selección SOLO con validation
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
        "kernel": kernel,
        "C": C,
        "gamma": gamma,
        "accuracy_validation": accuracy,
        "precision_macro_validation": precision,
        "recall_macro_validation": recall,
        "f1_macro_validation": f1
    })

    print(
        f"kernel={kernel:<6} | "
        f"C={C:<4} | "
        f"gamma={str(gamma):<5} | "
        f"Accuracy={accuracy:.4f} | "
        f"F1 macro={f1:.4f}"
    )


# ============================================================
# 6. COMPARAR CONFIGURACIONES
# ============================================================

resultados_df = pd.DataFrame(resultados)

resultados_df = resultados_df.sort_values(
    by="f1_macro_validation",
    ascending=False
).reset_index(drop=True)

resultados_df.to_csv(
    RESULTS_PATH / "svm_results.csv",
    index=False
)

print("\n=== MEJORES CONFIGURACIONES SVM ===")

print(
    resultados_df.to_string(index=False)
)


mejor = resultados_df.iloc[0]

print("\n=== MEJOR SVM SEGÚN VALIDATION ===")

print(f"kernel: {mejor['kernel']}")
print(f"C: {mejor['C']}")
print(f"gamma: {mejor['gamma']}")

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
    "results/svm_results.csv"
)