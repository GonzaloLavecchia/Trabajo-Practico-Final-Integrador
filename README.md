# Churn en telecomunicaciones

Trabajo Final Integrador del Diplomado en Ciencia de Datos y Análisis Avanzado.

Autores: Ana Espinosa, Micaela Venezia, Gonzalo Lavecchia, Nahuel Jimenez y Emiliano Ezequiel Greco Vairo.

## Objetivo

Clasificar la etiqueta observada de churn y evaluar un ranking para contactar al 20% de los clientes. La foto estática no permite predecir una fecha de baja ni un próximo período de facturación. Los resultados son un ejercicio retrospectivo y no prueban eficacia causal de retención.

## Ejecución local

Requiere Python 3.11 o 3.12 de 64 bits, conexión a internet en la primera ejecución y las librerías de requirements.txt. La ejecución de referencia utilizó Python 3.12.14 en Windows.

```bash
python -m venv .venv
```

Activar el entorno según el sistema operativo. Luego ejecutar:

```bash
python -m pip install -r requirements.txt
python analisis_churn.py
```

No se requiere una aplicación desplegada. El script descarga el CSV de IBM si falta y verifica su SHA-256 antes de entrenar. Los archivos se generan en resultados/. Una nueva ejecución reemplaza esos resultados. Los últimos decimales pueden variar por plataforma o librerías.

## Ejecución en Colab

1. Abrir Google Colab e importar Churn_Telecomunicaciones.ipynb desde la pestaña Subir.
2. Ejecutar la primera celda para instalar las versiones fijadas.
3. Si Colab ya había importado numpy o pandas, reiniciar el entorno de ejecución después de instalar.
4. Ejecutar las celdas siguientes. El notebook contiene el mismo experimento completo del script y descarga el archivo público. En Colab se escribe en /content/proyecto_churn/.
5. Guardar una copia en Drive y habilitar la visualización para el docente. Copiar el enlace real al envío académico.

## Publicación en GitHub

Crear un repositorio público desde la cuenta del equipo y subir el notebook, script, requirements.txt, README.md, .gitignore y resultados/. Si se desea, agregar el informe PDF. El archivo de datos no se incluye. Verificar que el repositorio sea accesible sin iniciar sesión y agregar su enlace real al envío. Este paquete no crea cuentas, repositorios públicos ni enlaces de Colab.

## Diseño del experimento

- Partición estratificada 80/20 con semilla 42, conservada en resultados/particion.csv.
- EDA de relaciones con la etiqueta únicamente sobre entrenamiento.
- Validación cruzada estratificada de cinco folds compartidos por todas las variantes.
- Pipelines con imputación, escalado y codificación ajustados en la porción de entrenamiento de cada fold.
- Regresión logística, Random Forest, XGBoost, pesos de clase, SMOTENC y ablación de TotalCharges.
- SMOTENC se aplica después de separar la prueba y exclusivamente al entrenamiento de cada fold. Validación y prueba nunca se remuestrean.
- Selección por Precision@20% media. Dentro de 1 punto porcentual del mejor valor se prefiere LR sin remuestreo y se desempata por AP. Es una regla de simplicidad; no representa una prueba de equivalencia estadística.
- Umbral auxiliar seleccionado con predicciones out-of-fold, sin tocar la prueba. Se usa para informar el criterio de recall y precision de la preentrega, no para el cupo del 20%.
- El modelo y el umbral se registran en decision_previa.json antes de calcular las métricas de prueba.
- Precision@20% se calcula con k = ceil(0.2 * n), orden descendente estable del score y proporción de etiquetas positivas entre los primeros k. El orden de entrada resuelve empates sin consultar el target. AP usa average_precision_score, no integración trapezoidal de PR.
- Bootstrap percentil con 2.000 remuestras de prueba y modelo fijo. El intervalo no incorpora variabilidad de entrenamiento ni cambios temporales.

## Resultados de referencia

Modelo elegido: regresión logística C=1. En prueba: 1.409 clientes, 374 casos de churn, 282 seleccionados y 186 aciertos dentro del grupo seleccionado.

| Métrica | Valor |
| --- | ---: |
| Precision@20% | 0.659574 |
| Recall@20% | 0.497326 |
| Lift@20% | 2.484867 |
| ROC AUC | 0.840887 |
| Average Precision | 0.631313 |
| Precision con umbral 0.285 | 0.514337 |
| Recall con umbral 0.285 | 0.767380 |

La ganancia de cinco puntos de F1 frente al baseline planteada en la preentrega no se alcanzó. Las comparaciones de CV se utilizan para seleccionar, no como estimaciones imparciales después de la búsqueda. El holdout es la evaluación final del modelo elegido.

## Archivos de resultados

- resultados.json: resultados completos, control de calidad, EDA, auditoría de grupos, versiones y checksum.
- comparacion_cv.csv y metricas_folds.csv: todos los experimentos y sus folds.
- particion.csv: índice de fila original base cero y pertenencia a train o test.
- predicciones_test.csv y predicciones_oof.csv: predicciones para recalcular métricas.
- decision_previa.json: criterio de selección y umbral, escrito antes de evaluar test.

gender y SeniorCitizen se usan solo para auditoría descriptiva. Su exclusión no garantiza ausencia de sesgos. No publicar información de clientes reales ni identificadores operativos.

## Datos y atribución

Fuente primaria: https://github.com/IBM/telco-customer-churn-on-icp4d

Archivo: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

SHA-256: 16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91

El repositorio de IBM publica licencia Apache 2.0 para su code pattern. Este paquete contiene una implementación propia del experimento y no redistribuye el CSV ni asigna licencias adicionales al material de terceros. Consultar los términos de la fuente para otros usos.

Documentación metodológica: scikit-learn common_pitfalls.html y average_precision_score; imbalanced-learn common_pitfalls.html y SMOTENC; IBM CRISP-DM Help Overview. Los enlaces completos están en el informe.
