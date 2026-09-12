from pathlib import Path
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd


# ============================================================
# RUTAS
# ============================================================

BASE_PATH = Path(__file__).resolve().parent.parent

RAW_PATH = BASE_PATH / "data" / "raw" / "d02_processed_data"
PROCESSED_PATH = BASE_PATH / "data" / "processed"

PROCESSED_PATH.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURACIÓN DEL DATASET
# ============================================================

CANALES = ["f1", "f2", "f3", "f4", "f5", "pitch3"]

# La actividad 014 fue excluida durante el análisis previo
ACTIVIDADES_EXCLUIDAS = {"014"}

filas = []


# ============================================================
# 1. EXTRAER FEATURES
# ============================================================

print("\n=== EXTRACCIÓN DE FEATURES ===\n")

for i in range(16):

    actividad = f"{i:03d}"

    if actividad in ACTIVIDADES_EXCLUIDAS:
        print(f"Actividad {actividad}: EXCLUIDA")
        continue

    archivo = RAW_PATH / f"{actividad}_2.npy"

    if not archivo.exists():
        print(f"ADVERTENCIA: no se encontró {archivo.name}")
        continue

    datos = np.load(archivo)

    print(
        f"Actividad {actividad}: "
        f"{datos.shape[0]} muestras | shape {datos.shape}"
    )

    for muestra in datos:

        fila = {}

        for indice_canal, nombre_canal in enumerate(CANALES):

            valores = muestra[:, indice_canal]

            fila[f"{nombre_canal}_mean"] = np.mean(valores)
            fila[f"{nombre_canal}_std"] = np.std(valores)
            fila[f"{nombre_canal}_min"] = np.min(valores)
            fila[f"{nombre_canal}_max"] = np.max(valores)

        fila["activity"] = actividad

        filas.append(fila)


# ============================================================
# 2. CREAR DATAFRAME
# ============================================================

df = pd.DataFrame(filas)

print("\n=== DATASET EXTRAÍDO ===")
print(f"Muestras: {len(df)}")
print(f"Columnas: {len(df.columns)}")
print(f"Features: {len(df.columns) - 1}")
print(f"Clases: {df['activity'].nunique()}")

print("\nClases encontradas:")
print(sorted(df["activity"].unique()))

print("\nMuestras por clase:")
print(df["activity"].value_counts().sort_index())


from sklearn.model_selection import train_test_split

feature_columns = [
    columna
    for columna in df.columns
    if columna != "activity"
]

print("\n=== LIMPIEZA DEL DATASET ===")

zero_mask = (df[feature_columns] == 0).all(axis=1)

print(f"Muestras completamente en cero: {zero_mask.sum()}")

df_clean = df.loc[~zero_mask].copy()

print(
    "Muestras después de eliminar señales en cero:",
    len(df_clean)
)

duplicados_antes = df_clean.duplicated().sum()

print(
    "Duplicados exactos adicionales:",
    duplicados_antes
)

df_clean = df_clean.drop_duplicates().reset_index(drop=True)

print(
    "Muestras después de eliminar duplicados:",
    len(df_clean)
)

conflictos = (
    df_clean.groupby(feature_columns, dropna=False)["activity"]
    .nunique()
)

conflictos = conflictos[conflictos > 1]

print(
    "Conflictos de etiqueta después de la limpieza:",
    len(conflictos)
)

if len(conflictos) > 0:
    print(
        "\nATENCIÓN: todavía existen features idénticas "
        "asociadas a actividades diferentes."
    )
    print(
        "No se realizará el split hasta revisar estos casos."
    )
    raise ValueError(
        "Existen conflictos de etiqueta después de la limpieza."
    )

print("\n=== DATASET LIMPIO ===")
print(f"Muestras finales: {len(df_clean)}")
print(f"Features: {len(feature_columns)}")
print(f"Clases: {df_clean['activity'].nunique()}")

print("\nMuestras finales por clase:")
print(df_clean["activity"].value_counts().sort_index())

output_file = PROCESSED_PATH / "rehab_features.csv"

df_clean.to_csv(
    output_file,
    index=False
)

print("\nDataset limpio guardado en:")
print(output_file)

train_df, temp_df = train_test_split(
    df_clean,
    test_size=0.30,
    stratify=df_clean["activity"],
    random_state=42
)

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["activity"],
    random_state=42
)

SPLITS_PATH = BASE_PATH / "data" / "splits"
SPLITS_PATH.mkdir(parents=True, exist_ok=True)

train_df.to_csv(
    SPLITS_PATH / "train.csv",
    index=False
)

validation_df.to_csv(
    SPLITS_PATH / "validation.csv",
    index=False
)

test_df.to_csv(
    SPLITS_PATH / "test.csv",
    index=False
)

print("\n=== SPLIT DEFINITIVO ===")

print(
    f"Train: {len(train_df)} "
    f"({len(train_df) / len(df_clean):.2%})"
)

print(
    f"Validation: {len(validation_df)} "
    f"({len(validation_df) / len(df_clean):.2%})"
)

print(
    f"Test: {len(test_df)} "
    f"({len(test_df) / len(df_clean):.2%})"
)

print("\nClases en train:")
print(train_df["activity"].value_counts().sort_index())

print("\nClases en validation:")
print(validation_df["activity"].value_counts().sort_index())

print("\nClases en test:")
print(test_df["activity"].value_counts().sort_index())