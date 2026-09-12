# REHAB: selección, configuración y entrenamiento de modelos

Comparación de KNN, Decision Tree, Random Forest y SVM para clasificar actividades de rehabilitación mediante características numéricas del dataset REHAB.

## Integrantes

- Héctor Kenneth Ramos Velázquez — A01642284
- Elisheba Hannaí Trejo Leyva — A01736967

## Datos

Se utilizaron 24 características: media, desviación estándar, mínimo y máximo de los canales `f1`, `f2`, `f3`, `f4`, `f5` y `pitch3`.

La variable objetivo es `activity`. Se consideran 15 clases (`000`–`013` y `015`), conservando la exclusión de `014` acordada a partir del EDA y la recomendación del profesor.

Los cuatro modelos utilizan las mismas particiones:

| Archivo | Muestras | Uso |
|---|---:|---|
| `data/train.csv` | 2,534 | Entrenamiento |
| `data/validation.csv` | 543 | Selección de configuraciones |
| `data/test.csv` | 543 | Evaluación final |

Total: **3,620 muestras**, distribuidas en 70% entrenamiento, 15% validación y 15% prueba.

Se verificó la ausencia de valores faltantes, infinitos, filas duplicadas internas y vectores de características idénticos compartidos entre particiones.

## Metodología

- Implementación mediante scikit-learn.
- KNN y SVM utilizan estandarización ajustada exclusivamente con entrenamiento.
- Decision Tree y Random Forest utilizan las características sin estandarizar.
- Las configuraciones se seleccionan por **F1 macro de validación**.
- En empates exactos se conserva la primera configuración evaluada.
- La evaluación final utiliza las configuraciones seleccionadas, entrenadas únicamente con `train.csv`.
- Los resultados de test no se utilizan para modificar parámetros ni seleccionar nuevamente el modelo.

## Configuraciones seleccionadas

| Modelo | Configuración |
|---|---|
| Decision Tree | `max_depth=None`, `min_samples_leaf=1` |
| Random Forest | `n_estimators=100`, `max_depth=None`, `min_samples_leaf=1`, `max_features="sqrt"` |
| KNN | `n_neighbors=1`, `weights="uniform"`, `metric="manhattan"` |
| SVM | `kernel="rbf"`, `C=10`, `gamma="scale"` |

Se probaron 16 configuraciones de Decision Tree, 12 de Random Forest, 20 de KNN y 9 de SVM.

## Resultados

Métricas expresadas en porcentaje:

| Modelo | F1 macro de validación | Accuracy de test | F1 macro de test |
|---|---:|---:|---:|
| **Random Forest** | **90.42** | **90.06** | **89.99** |
| KNN | 83.42 | 82.87 | 82.69 |
| Decision Tree | 74.66 | 74.03 | 74.04 |
| SVM | 72.33 | 70.72 | 70.59 |

**Random Forest fue seleccionado por su mayor F1 macro de validación.** También obtuvo el mejor desempeño global en prueba entre las configuraciones evaluadas.

La comparación corresponde a una única partición. Los CSV no incluyen identificadores de participante o sesión, por lo que no permiten verificar independencia entre participantes.

## Organización

- `data/`: particiones compartidas.
- `hector/`: scripts de Decision Tree y Random Forest.
- `elisheba/`: scripts de KNN y SVM.
- `results/`: métricas y configuraciones seleccionadas.
- `docs/`: reporte en PDF con metodología, resultados y matrices de confusión.
- `evaluate_models.py`: evaluación final de los cuatro modelos.
- `requirements.txt`: versiones de las dependencias.

KNN, SVM y Random Forest reutilizan funciones de `hector/decision_tree_model.py`; se debe conservar esta estructura.

## Instalación

Desde la raíz del proyecto, en PowerShell de Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Ejecución

Entrenar y comparar configuraciones en validación:

```powershell
.\.venv\Scripts\python.exe hector/decision_tree_model.py
.\.venv\Scripts\python.exe hector/random_forest_model.py
.\.venv\Scripts\python.exe elisheba/knn_model.py
.\.venv\Scripts\python.exe elisheba/svm_model.py
```

Cada script guarda en su carpeta de `results/`:

- `validation_metrics.csv`
- `best_configuration.json`
- `validation_confusion_matrix.png`

Random Forest también genera `feature_importances.csv`.

Con las configuraciones fijadas, ejecutar la evaluación final:

```powershell
.\.venv\Scripts\python.exe evaluate_models.py
```

Este script necesita los cuatro archivos `best_configuration.json`. Genera en `results/final_test/` la tabla `test_metrics.csv`, las predicciones y las matrices de confusión en CSV y PNG.

Las ejecuciones vuelven a generar los archivos de resultados correspondientes.

## Documentación

El [reporte en PDF está en la carpeta docs](docs/). Incluye la justificación de los modelos, las configuraciones probadas, las métricas, las matrices de confusión y las conclusiones.