# Predicción mensual de nuevas maternidades equivalentes en un hospital

## Proyecto educativo de regresión lineal múltiple aplicada a planificación de personal

Este repositorio desarrolla un proyecto completo de **predicción cuantitativa aplicada a la gestión de recursos humanos hospitalarios**.

El objetivo es construir, evaluar y desplegar un modelo capaz de estimar el volumen mensual esperado de **nuevas maternidades equivalentes**. La idea no es predecir decisiones individuales, sino anticipar un fenómeno agregado que puede afectar a la planificación de sustituciones, la organización asistencial y la previsión presupuestaria.

El proyecto sigue un flujo completo de trabajo:

1. construcción y limpieza del dataset;
2. análisis exploratorio;
3. creación de variables predictoras;
4. entrenamiento de modelos de regresión lineal múltiple;
5. comparación contra baselines simples;
6. despliegue de una app interactiva con Streamlit.

> **Nota sobre los datos:** los datos incluidos en este repositorio son ficticios y se utilizan únicamente con fines educativos e ilustrativos. No corresponden a datos reales de ninguna institución ni contienen información personal o administrativa identificable.

---

## 1. Pregunta de predicción

La pregunta principal del proyecto es:

> ¿Podemos anticipar el volumen mensual de nuevas bajas maternales equivalentes de trabajadoras en un hospital utilizando información agregada de plantilla, estabilidad contractual, estructura familiar aproximada y riesgo durante el embarazo?

La predicción se realiza a escala **mensual**, porque esta frecuencia es útil para la planificación operativa y presupuestaria.

---

## 2. Variable objetivo

La variable objetivo es:

```text
mat_eq_nuevas_mes
```

Representa las **nuevas maternidades equivalentes mensuales**.

Se calcula como:

```text
mat_eq_nuevas_mes = dias_mat_nuevas_mes / dias_naturales_mes
```

La lógica es sencilla: una maternidad que empieza al inicio del mes genera más carga mensual que una maternidad que empieza al final.

Por ejemplo, si una baja maternal empieza el día 16 de un mes de 30 días, aporta 15 días dentro de ese mes:

```text
15 / 30 = 0,5
```

Por eso esta variable es más informativa que un simple recuento de maternidades nuevas. Mide mejor el impacto mensual equivalente del fenómeno.

---

## 3. Variables explicativas

El dataset utiliza variables agregadas y mensuales. Las principales son:

| Variable | Descripción |
|---|---|
| `MES` | Mes de referencia. |
| `mat_eq_nuevas_mes` | Variable objetivo: nuevas maternidades equivalentes mensuales. |
| `ppef_mujeres_25_40` | Plantilla equivalente de mujeres de 25 a 40 años. Aproxima el tamaño de la población expuesta. |
| `pct_indef_25_40` | Porcentaje de mujeres de 25 a 40 años con contrato indefinido. Aproxima estabilidad laboral. |
| `pct_amb_fill_25_40` | Porcentaje de mujeres de 25 a 40 años con al menos un hijo menor declarado en nómina. Es una proxy administrativa de estructura familiar. |
| `RE_ponderado_lag1` | Indicador ponderado de riesgo durante el embarazo con un mes de retardo. |
| `RE_ponderado_lag2` | Indicador ponderado de riesgo durante el embarazo con dos meses de retardo. |
| `RE_ponderado_lag3` | Indicador ponderado de riesgo durante el embarazo con tres meses de retardo. |
| `mat_eq_lag1` | Valor de la variable objetivo en el mes anterior. |
| `mat_eq_lag12` | Valor de la variable objetivo en el mismo mes del año anterior. |
| `mes` | Número de mes, usado como variable de estacionalidad. |

La variable de riesgo durante el embarazo se considera especialmente relevante porque puede funcionar como una **señal adelantada**: algunas situaciones de riesgo durante el embarazo terminan posteriormente en una baja maternal.

---

## 4. Flujo del proyecto

El proyecto se organiza en tres notebooks principales.

### 4.1 Notebook 01 — Data Cleaning & EDA

Archivo:

```text
notebooks/01_data_cleaning_eda.ipynb
```

Este notebook prepara los datos y realiza el análisis exploratorio.

Incluye:

- carga del archivo inicial;
- conversión de `MES` a formato fecha;
- ordenación temporal;
- revisión de tipos, nulos y estadísticas descriptivas;
- visualización de la evolución de la variable objetivo;
- comparación visual con `ppef_mujeres_25_40` y `RE_ponderado`;
- creación de variables retardadas;
- análisis de correlación Pearson y Spearman;
- análisis de colinealidad mediante matriz de correlación y VIF;
- exportación del dataset limpio para modelado.

El resultado principal de este notebook es:

```text
data/df_model_full.csv
```

Este archivo contiene todas las variables candidatas limpias y preparadas para el modelado.

---

### 4.2 Notebook 02 — Modeling

Archivo:

```text
notebooks/02_modeling.ipynb
```

Este notebook construye y compara modelos de **regresión lineal múltiple**.

El proceso incluye:

- carga de `df_model_full.csv`;
- eliminación de los primeros meses sin histórico suficiente para los retardos;
- división temporal train/test;
- generación controlada de combinaciones de variables;
- entrenamiento de modelos con `scikit-learn LinearRegression`;
- entrenamiento de modelos con `statsmodels OLS`;
- comparación de modelos mediante MAE, RMSE, R² y sMAPE;
- selección prudente del mejor modelo;
- análisis de predicciones, residuos, coeficientes y colinealidad del modelo final.

La búsqueda de modelos no consiste en probar cualquier combinación sin criterio. Se limita el tamaño de los modelos y se evitan combinaciones redundantes, como incluir varios retardos muy similares del mismo indicador.

---

### 4.3 Notebook 03 — Baselines and Model Comparison

Archivo:

```text
notebooks/03_baselines_and_model_comparison.ipynb
```

Este notebook responde a la pregunta clave:

> ¿El modelo de regresión lineal múltiple mejora realmente reglas simples de predicción?

Para ello compara el mejor modelo de RLM contra varios baselines:

| Baseline | Descripción |
|---|---|
| Media histórica global | Predice siempre el promedio histórico del entrenamiento. |
| Media histórica por mes | Predice el promedio histórico de ese mismo mes del año. |
| Último valor observado | Usa como predicción el valor del mes anterior. |
| Mismo mes del año anterior | Usa como predicción el valor observado 12 meses antes. |
| Media móvil de 3 meses | Usa el promedio de los tres meses anteriores. |

Esta comparación es fundamental. Un modelo predictivo sólo tiene sentido si mejora alternativas sencillas, transparentes y fáciles de implementar.

---

## 5. Criterio de evaluación

La métrica principal del proyecto es:

```text
MAE — Mean Absolute Error
```

El MAE se interpreta en las mismas unidades que la variable objetivo. Por ejemplo, un MAE de 1,2 significa que el modelo se equivoca, de media, en unas 1,2 maternidades equivalentes mensuales.

También se calculan:

```text
RMSE
R²
sMAPE
```

La selección del modelo no se basa sólo en el menor error. También se tienen en cuenta:

- parsimonia;
- interpretabilidad;
- estabilidad;
- ausencia de colinealidad severa;
- utilidad operativa;
- mejora frente a baselines.

---

## 6. Aplicación Streamlit

Archivo:

```text
app/app.py
```

El proyecto incluye una aplicación interactiva desarrollada con **Streamlit**.

La app permite:

- cargar el dataset de modelado;
- visualizar la serie histórica de `mat_eq_nuevas_mes`;
- consultar métricas descriptivas básicas;
- ver un gráfico de correlaciones;
- comparar el modelo contra baselines;
- introducir escenarios manuales;
- obtener una predicción mensual esperada.

La aplicación utiliza una versión desplegable del modelo basada en:

```text
StandardScaler + LinearRegression
```

y permite simular escenarios introduciendo valores para:

```text
ppef_mujeres_25_40
RE_ponderado_lag1
mes
```

---

## 7. Cómo ejecutar el proyecto

### 7.1 Clonar el repositorio

```bash
git clone https://github.com/tu_usuario/maternity-leave-forecasting-mlr.git
cd maternity-leave-forecasting-mlr
```

### 7.2 Crear entorno e instalar dependencias

```bash
pip install -r requirements.txt
```

### 7.3 Ejecutar los notebooks

Orden recomendado:

```text
notebooks/01_data_cleaning_eda.ipynb
notebooks/02_modeling.ipynb
notebooks/03_baselines_and_model_comparison.ipynb
```

### 7.4 Reproducir y ejecutar la aplicación Streamlit

La app se encuentra en:

```text
app/app.py
```

Para reproducirla en local, el repositorio debe mantener esta estructura:

```text
maternity-leave-forecasting-mlr/
│
├── app/
│   └── app.py
│
├── data/
│   └── df_model_full.csv
│
└── requirements.txt
```

La app espera encontrar el dataset de modelado en:

```text
data/df_model_full.csv
```

Si este archivo todavía no existe, ejecuta primero los notebooks en el orden recomendado:

```text
notebooks/01_data_cleaning_eda.ipynb
notebooks/02_modeling.ipynb
notebooks/03_baselines_and_model_comparison.ipynb
```

Después, desde la raíz del repositorio, instala las dependencias:

```bash
pip install -r requirements.txt
```

Finalmente, lanza la app con:

```bash
streamlit run app/app.py
```

La aplicación se abrirá normalmente de forma automática en el navegador. Si no se abre, entra en:

```text
http://localhost:8501
```

Para detener la app y volver a la terminal, pulsa:

```text
Ctrl + C
```

#### Qué hace la app

La aplicación Streamlit reproduce la parte desplegable del proyecto. Permite:

- cargar `data/df_model_full.csv`;
- convertir `MES` a formato fecha;
- entrenar un pipeline desplegable con `StandardScaler + LinearRegression`;
- comparar el modelo contra baselines simples;
- visualizar la serie histórica de `mat_eq_nuevas_mes`;
- mostrar un mapa de correlación de Pearson;
- simular escenarios manuales usando:

```text
ppef_mujeres_25_40
RE_ponderado_lag1
mes
```

La salida es una predicción mensual esperada de nuevas maternidades equivalentes.

#### Solución de problemas frecuentes

Si Streamlit no encuentra el dataset, revisa que el archivo exista aquí:

```text
data/df_model_full.csv
```

y que el comando se esté ejecutando desde la raíz del repositorio, no desde dentro de la carpeta `app/`.

Si faltan paquetes, reinstala las dependencias:

```bash
pip install -r requirements.txt
```

---

## 8. Estructura del repositorio

```text
maternity-leave-forecasting-mlr/
│
├── README.md
├── README_ESP.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── dataframe.xlsx
│   └── df_model_full.csv
│
├── notebooks/
│   ├── 01_data_cleaning_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_baselines_and_model_comparison.ipynb
│
├── app/
│   └── app.py
│
└── outputs/
    ├── reports/
    └── model/
```

### Archivos principales

| Archivo | Función |
|---|---|
| `data/dataframe.xlsx` | Dataset ficticio inicial. |
| `data/df_model_full.csv` | Dataset limpio usado para modelado. |
| `notebooks/01_data_cleaning_eda.ipynb` | Limpieza, transformación y EDA. |
| `notebooks/02_modeling.ipynb` | Construcción y selección de modelos RLM. |
| `notebooks/03_baselines_and_model_comparison.ipynb` | Comparación contra baselines y evaluación final. |
| `app/app.py` | Aplicación Streamlit interactiva. |
| `requirements.txt` | Dependencias del proyecto. |
| `.gitignore` | Exclusiones de archivos locales, checkpoints y salidas no necesarias. |

---

## 9. Resultados esperados

Este proyecto no busca eliminar la incertidumbre. Busca reducirla de forma medible.

Un modelo se considerará útil si:

- mejora claramente los baselines simples;
- mantiene un error medio aceptable;
- utiliza pocas variables;
- sus resultados son interpretables;
- puede integrarse en una herramienta de apoyo a la planificación.

En la comparación final, si el modelo de regresión lineal múltiple reduce el MAE frente al mejor baseline, puede considerarse una primera herramienta válida para apoyar la predicción mensual de nuevas maternidades equivalentes.

---

## 10. Limitaciones

El proyecto tiene limitaciones importantes:

- trabaja con datos agregados mensuales;
- no predice decisiones individuales;
- la variable `pct_amb_fill_25_40` es una proxy administrativa imperfecta;
- la relación entre riesgo durante el embarazo y maternidad puede variar con el tiempo;
- los resultados dependen del periodo histórico disponible;
- el modelo debe revisarse y reentrenarse periódicamente.

Además, el dataset publicado es ficticio, por lo que el objetivo del repositorio es metodológico y educativo.

---

## 11. Próximas mejoras

Posibles extensiones del proyecto:

- validación rolling-origin;
- comparación con Ridge y Lasso;
- intervalos de predicción;
- mejora del indicador `RE_ponderado`;
- modelado del arrastre de maternidades activas;
- actualización mensual automatizada;
- generación de informes;
- despliegue de la app en Streamlit Community Cloud.

---

## Conclusión

Este proyecto muestra cómo una técnica sencilla e interpretable como la **regresión lineal múltiple** puede aplicarse a un problema realista de gestión hospitalaria.

La predicción mensual de nuevas maternidades equivalentes no elimina la incertidumbre, pero ayuda a pensarla de forma más estructurada. Convierte información administrativa dispersa en una estimación útil para planificar, comparar escenarios y tomar mejores decisiones.

En ese sentido, el valor del proyecto no está sólo en el modelo final, sino en el proceso: definir bien la variable objetivo, construir el dataset, comparar alternativas, medir el error y contrastar siempre el resultado contra baselines simples.
