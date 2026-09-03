# TP1 — Machine Learning: Telco Customer Churn

Trabajo Práctico 1 de **Tecnología Digital VI – Inteligencia Artificial** (Licenciatura en Tecnología Digital, Universidad Torcuato Di Tella, 2do semestre 2026).

Autor: **Luciano Jacofsky**

## Objetivo

Recorrer el ciclo de vida completo de un problema de Machine Learning sobre el dataset [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn): preparación de los datos, prevención de *data leakage*, estrategia de validación y elección de métricas orientadas al negocio.

El trabajo se divide en dos partes:

- **Parte I** — Predicción de `Churn` (abandono de clientes): EDA, preprocesamiento, validación cruzada, modelo base, balanceo de clases y análisis de impacto económico.
- **Parte II** — Reformulación del problema: se abandona `Churn` y se plantea un nuevo target de clasificación binaria a partir del mismo dataset.

## Dataset

- **Archivo:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Filas / columnas:** 7.043 clientes, 21 variables.
- **Target original:** `Churn` (`Yes`/`No`), desbalanceado (26,5% / 73,5%).

## Estructura del repositorio

```
.
├── TP1_Telco_Churn_seed7.ipynb   # Notebook principal (entrega)
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── README.md
```

## Cómo correrlo

Requiere Python 3.10+.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

Abrir `TP1_Telco_Churn_seed7.ipynb` en Jupyter/VS Code/Colab y ejecutar todas las celdas en orden (`Restart & Run All`). El CSV debe estar en el mismo directorio que el notebook.

## Contenido del notebook

**Parte I — Churn**

1. Análisis exploratorio: estadísticas descriptivas, distribución de clases y visualizaciones. Se detecta que `TotalCharges` está mal tipada (texto en vez de numérico) y se corrige.
2. Preprocesamiento: imputación de `TotalCharges` (clientes con `tenure=0`), colapso de categorías redundantes, codificación con `OneHotEncoder` dentro de un `ColumnTransformer`/`Pipeline` para evitar leakage entre train y test.
3. Validación con `StratifiedKFold` (K=5), justificando su uso frente a un Holdout simple dado el desbalance de clases.
4. Modelo base: se diagnostica overfitting en el árbol sin restricciones (99,84% de Accuracy en train vs. 72,54% en validación) y se ajustan `max_depth` y `min_samples_leaf` con `GridSearchCV` optimizando F1.
5. Balanceo de clases (oversampling vs. undersampling) evaluado con múltiples particiones y con Average Precision para verificar si la mejora es real.
6. Análisis de negocio: costo de Falsos Positivos vs. Falsos Negativos, punto de equilibrio de una campaña de descuentos, y elección del umbral de decisión por Cross-Validation y por costo esperado.

**Parte II — Reformulando el problema**

Primer intento (`MonthlyCharges` alto/bajo) descartado al detectar que el target era reconstruible casi exactamente a partir de otras columnas (`TotalCharges / tenure`), es decir, no era un problema predictivo genuino. Se lo reemplaza por predecir `Contract == "Month-to-month"`, un target sin esa dependencia estructural.

## Resultados principales

| Modelo / Etapa | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Naive (clase mayoritaria) | 73,46% | – | – | – |
| Árbol sin restricciones (CV) | 72,54% | – | – | 47,76% |
| Árbol tuneado (GridSearchCV) — test | 80,77% | 63,73% | 63,90% | 63,82% |
| + Oversampling — test | 75,59% | 52,61% | 80,75% | 63,71% |
| Parte II — `MonthlyCharges` alto/bajo (descartado, target leakeado) | 98,08% | 97,99% | 98,13% | 98,06% |
| Parte II — `Contract` mes a mes (target final) | 78,78% | – | – | 80,47% |

## Conclusiones

- El Accuracy no es una métrica confiable en este problema: el árbol sin ajustar queda por debajo del modelo naive. Ajustar la complejidad del árbol (`GridSearchCV`) fue más determinante que cualquier técnica de balanceo.
- El balanceo de clases (oversampling) no mejora la capacidad de discriminación del modelo (Average Precision similar o peor); el mismo efecto se logra moviendo el umbral de decisión, de forma más barata y controlada.
- Un Falso Negativo cuesta ~8 veces más que un Falso Positivo bajo los supuestos de negocio adoptados, pero la decisión final depende más de qué tan efectivo sea el descuento en retener clientes que del modelo elegido.
- Un resultado "demasiado bueno" (F1 ≈ 98%) es señal de alerta, no de éxito: en la Parte II reveló una relación estructural entre variables (leakage de facto) que fue necesario detectar y corregir.

## Posibles mejoras futuras

- Comparar contra Random Forest / Gradient Boosting.
- Sumar una Regresión Logística con variables escaladas.
- Reemplazar los supuestos económicos por datos reales o una prueba piloto A/B.
- Elegir el umbral de decisión directamente por costo esperado en vez de por F1.
