# Predicción mensual de nuevas maternidades equivalentes en un hospital

## Proyecto de regresión lineal múltiple para anticipar nuevas bajas maternales de trabajadoras

Este repositorio desarrolla un proyecto completo de **predicción cuantitativa aplicada a la gestión de recursos humanos hospitalarios**.

El objetivo es construir un modelo de **regresión lineal múltiple** capaz de estimar, con frecuencia mensual, el volumen esperado de **nuevas bajas maternales equivalentes** de trabajadoras en un hospital.

La idea central no es predecir decisiones individuales, sino anticipar un fenómeno agregado que puede afectar a la planificación de personal, la cobertura asistencial y la previsión de gasto. Para ello se combinan variables demográficas, laborales y administrativas que pueden estar relacionadas con la aparición futura de nuevas maternidades.

El proyecto reproduce las fases habituales de un flujo de predicción cuantitativa: definición del problema, construcción del dataset, análisis exploratorio, selección de variables, entrenamiento del modelo, evaluación y despliegue en una aplicación interactiva.

> **Nota sobre los datos:** los datos incluidos en este repositorio son ficticios y se utilizan únicamente con fines educativos e ilustrativos. No corresponden a datos reales de ninguna institución ni contienen información personal o administrativa identificable.

---

## 1. Definir el objetivo del proyecto de predicción cuantitativa

Todo proyecto predictivo debe empezar con una pregunta concreta.

En este caso, la pregunta es:

> ¿Podemos prever el volumen mensual de nuevas bajas maternales equivalentes de trabajadoras en un hospital utilizando información agregada de plantilla, estabilidad contractual, estructura familiar aproximada y bajas por riesgo durante el embarazo?

### Variable objetivo

La variable objetivo del proyecto es:

```text
mat_eq_nuevas_mes
```

Esta variable representa las **nuevas maternidades equivalentes mensuales**.

Se calcula como:

```text
mat_eq_nuevas_mes = dias_mat_nuevas_mes / dias_naturales_mes
```

Donde:

- `dias_mat_nuevas_mes` es la suma de días del mes aportados por maternidades que se han iniciado durante ese mismo mes.
- `dias_naturales_mes` es el número de días naturales del mes.

Por ejemplo, si una baja maternal empieza el día 16 de un mes de 30 días, aporta 15 días dentro de ese mes. Su valor equivalente será:

```text
15 / 30 = 0,5
```

La elección de esta variable permite capturar mejor el impacto operativo mensual que el simple recuento de maternidades nuevas. No pesa igual una maternidad que empieza el primer día del mes que otra que empieza el último día.

### Horizonte de predicción

El proyecto trabaja con una frecuencia **mensual**.

La previsión mensual es especialmente útil para:

- anticipar necesidades de cobertura,
- planificar sustituciones,
- estimar impacto presupuestario,
- detectar meses con mayor presión organizativa,
- mejorar la toma de decisiones en gestión de personal.

---

## 2. Contexto del problema y variables consideradas

Este proyecto utiliza conceptos propios de la gestión de personal en un hospital público. Para facilitar la comprensión, esta sección resume las principales variables utilizadas.

### Plantilla equivalente

La **plantilla equivalente** transforma días de contrato, días de baja o días de actividad en una medida comparable entre meses.

De forma general:

```text
plantilla_equivalente = dias_computables / dias_naturales_mes
```

Si una persona está contratada o de baja durante todo un mes, su valor equivalente será aproximadamente 1. Si sólo lo está medio mes, su valor equivalente será aproximadamente 0,5.

En este proyecto se utiliza una variable agregada de población expuesta:

```text
ppef_mujeres_25_40
```

Esta variable representa la plantilla equivalente de mujeres entre 25 y 40 años. Se utiliza como aproximación al tamaño de la población con mayor probabilidad de generar nuevas bajas maternales.

### Nuevas maternidades equivalentes

La variable `mat_eq_nuevas_mes` mide la carga equivalente generada por maternidades iniciadas en el mes.

No mide todas las maternidades activas, sino sólo la entrada mensual de nuevas bajas maternales. Esto permite analizar la incidencia del fenómeno, no el stock acumulado.

### Número de maternidades nuevas

La variable:

```text
n_maternidades_nuevas
```

indica el número bruto de bajas maternales iniciadas en el mes.

No será necesariamente una variable predictora, porque forma parte de la construcción de la variable objetivo, pero sí es útil para interpretar los resultados.

### Días de nuevas maternidades en el mes

La variable:

```text
dias_mat_nuevas_mes
```

recoge los días del mes generados por maternidades iniciadas en ese mismo mes.

Es el numerador de la variable objetivo.

### Porcentaje de trabajadoras indefinidas

La variable:

```text
pct_indef_25_40
```

representa el porcentaje de mujeres de 25 a 40 años con contrato indefinido.

La hipótesis es que la estabilidad laboral puede estar relacionada con decisiones vitales y familiares. No se asume causalidad directa, pero se explora si esta variable ayuda a mejorar la predicción agregada.

### Porcentaje de trabajadoras con hijo menor declarado

La variable:

```text
pct_amb_fill_25_40
```

representa la proporción de mujeres de 25 a 40 años con al menos un hijo menor declarado en nómina, a partir de la información disponible en la retención de IRPF.

Debe interpretarse como una **proxy administrativa** de estructura familiar, no como una medición perfecta del número real de hijos. Puede ayudar a aproximar la composición familiar de la población objetivo.

### Riesgo durante el embarazo

El **riesgo durante el embarazo** es una situación administrativa previa a la maternidad que puede aparecer cuando el puesto de trabajo supone un riesgo para la trabajadora embarazada o para el feto.

En este proyecto se considera una señal adelantada relevante, porque muchas situaciones de riesgo durante el embarazo terminan posteriormente en una baja maternal.

Se han construido variables agregadas de riesgo activo al cierre del mes anterior:

```text
RE_activo_total_lag1
RE_activo_0_1m_lag1
RE_activo_1_2m_lag1
RE_activo_2_3m_lag1
RE_activo_3m_plus_lag1
```

Estas variables clasifican las situaciones activas de riesgo durante el embarazo según su antigüedad.

Además, se construye una variable sintética:

```text
RE_ponderado
```

que resume las cohortes de riesgo activo ponderando más aquellas que, por antigüedad, podrían estar más cerca de transformarse en una maternidad.

---

## 3. Adquisición, limpieza y transformación de los datos (ETL)

El dataset final se obtiene a partir de fuentes administrativas internas agregadas. El objetivo del proceso ETL es transformar registros individuales en una tabla mensual lista para modelar.

El notebook encargado de esta fase será:

```text
notebooks/01_data_cleaning_eda.ipynb
```

### Archivo de entrada

El dataset agregado se guardará en:

```text
data/df_noves_maternitats.csv
```

Cada fila representa un mes.

### Estructura esperada del dataset

El dataframe inicial incluye variables como:

```text
MES
ppef_mujeres_25_40
pct_indef_25_40
pct_amb_fill_25_40
n_maternidades_nuevas
dias_mat_nuevas_mes
mat_eq_nuevas_mes
RE_activo_total_lag1
RE_activo_0_1m_lag1
RE_activo_1_2m_lag1
RE_activo_2_3m_lag1
RE_activo_3m_plus_lag1
RE_ponderado
```

### Transformaciones principales

Durante la fase ETL se realizarán, como mínimo, los siguientes pasos:

1. Importar el archivo original.
2. Convertir la columna `MES` a formato fecha.
3. Ordenar cronológicamente el dataset.
4. Revisar valores nulos, duplicados y tipos de datos.
5. Verificar la coherencia de la variable objetivo.
6. Crear variables temporales auxiliares:

```text
mes
trimestre
año
```

7. Crear variables retardadas si la serie mensual es continua:

```text
mat_eq_lag1
mat_eq_lag12
RE_ponderado_lag1
RE_ponderado_lag2
RE_ponderado_lag3
```

8. Exportar el dataset limpio para modelado.


## 4. Fase EDA: análisis exploratorio de datos

El análisis exploratorio tiene como finalidad entender la estructura del dataset antes de entrenar el modelo.

Se desarrollará en:

```text
notebooks/01_data_cleaning_eda.ipynb
```

### 4.1 Revisión inicial

Se analizarán:

- dimensiones del dataset,
- rango temporal disponible,
- valores nulos,
- duplicados,
- tipos de datos,
- estadísticas descriptivas,
- distribución de la variable objetivo.

### 4.2 Evolución temporal

Se visualizará la evolución mensual de:

- `mat_eq_nuevas_mes`,
- `ppef_mujeres_25_40`,
- `pct_indef_25_40`,
- `pct_amb_fill_25_40`,
- `RE_ponderado`.

El objetivo es observar si existen tendencias, rupturas, cambios de nivel o meses atípicos.

### 4.3 Análisis de correlación lineal

Se calcularán matrices de correlación usando dos métodos:

#### Pearson

Mide la asociación lineal entre variables numéricas.

#### Spearman

Mide la asociación monotónica y es menos sensible a relaciones no estrictamente lineales o a valores extremos.

Se analizará la correlación entre la variable objetivo y las principales variables explicativas:

```text
ppef_mujeres_25_40
pct_indef_25_40
pct_amb_fill_25_40
RE_ponderado
RE_ponderado_lag1
RE_ponderado_lag2
RE_ponderado_lag3
```

### 4.4 Análisis de colinealidad

La regresión lineal múltiple requiere vigilar la colinealidad entre variables explicativas.

Para ello se utilizarán dos herramientas:

1. **Matriz de correlación entre predictores**
2. **VIF — Variance Inflation Factor**

El VIF permite detectar variables que aportan información muy redundante respecto a las demás. Si dos variables están muy correlacionadas, puede ser necesario eliminar una de ellas o comparar modelos alternativos.

Esto será especialmente importante en variables como:

```text
ppef_mujeres_25_40
pct_indef_25_40
pct_amb_fill_25_40
```

porque pueden estar relacionadas con la evolución estructural de la plantilla.

### 4.5 Feature importance con Random Forest

Aunque el modelo principal será una regresión lineal múltiple, se utilizará un modelo auxiliar de **Random Forest Regressor** para obtener una primera aproximación a la importancia relativa de las variables.

Este análisis no sustituye a la interpretación de la regresión lineal, pero puede ayudar a detectar qué variables contienen más señal predictiva.

Variables candidatas:

```text
ppef_mujeres_25_40
pct_indef_25_40
pct_amb_fill_25_40
RE_ponderado
RE_ponderado_lag1
RE_ponderado_lag2
RE_ponderado_lag3
mat_eq_lag1
mat_eq_lag12
mes
```

---

## 5. Construcción del modelo de predicción

La construcción y evaluación del modelo se realizará en:

```text
notebooks/02_modeling.ipynb
```

El objetivo es construir y comparar modelos de **regresión lineal múltiple** para predecir:

```text
mat_eq_nuevas_mes
```

### 5.1 Modelos a comparar

Se compararán dos métodos de regresión lineal múltiple en Python.

#### Modelo 1. Regresión lineal múltiple con `scikit-learn`

Este modelo estará orientado a la predicción práctica y al despliegue posterior en Streamlit.

Ventajas:

- integración sencilla en pipelines,
- facilidad para generar predicciones,
- compatibilidad con `joblib` o `pickle`,
- uso directo en la aplicación.

#### Modelo 2. Regresión lineal múltiple con `statsmodels` — OLS

Este modelo estará orientado a la interpretación estadística.

Ventajas:

- análisis de coeficientes,
- intervalos de confianza,
- significación estadística,
- diagnóstico básico del ajuste,
- mayor claridad para explicar el modelo.

### 5.2 Modelos candidatos

Se construirán varias especificaciones progresivas.

#### Modelo base estructural

```text
mat_eq_nuevas_mes
~ ppef_mujeres_25_40
+ pct_indef_25_40
+ pct_amb_fill_25_40
+ mes
```

#### Modelo con señal adelantada de riesgo embarazo

```text
mat_eq_nuevas_mes
~ ppef_mujeres_25_40
+ pct_indef_25_40
+ pct_amb_fill_25_40
+ RE_ponderado_lag2
+ mes
```

#### Modelo con inercia temporal

```text
mat_eq_nuevas_mes
~ ppef_mujeres_25_40
+ pct_amb_fill_25_40
+ RE_ponderado_lag2
+ mat_eq_lag1
+ mes
```

La selección final dependerá de la evaluación fuera de muestra y de la estabilidad de los coeficientes.

### 5.3 División train/test

Como se trata de una serie mensual, no se debe dividir el dataset de forma aleatoria.

Se utilizará una división temporal:

```text
train = primeros meses del histórico
test = últimos 12 o 18 meses
```

Esto permite evaluar el modelo en una situación más parecida al uso real: entrenar con el pasado y predecir meses futuros.

### 5.4 Métricas de evaluación

Se utilizarán varias métricas para comparar los modelos:

```text
MAE
RMSE
R²
MAPE o sMAPE, si procede
```

#### MAE

Error absoluto medio. Es fácil de interpretar porque está en las mismas unidades que la variable objetivo.

#### RMSE

Penaliza más los errores grandes. Es útil para detectar modelos que fallan mucho en algunos meses.

#### R²

Indica la proporción de variabilidad explicada, aunque no debe utilizarse como único criterio de selección.

#### MAPE / sMAPE

Puede ser útil, pero debe interpretarse con cuidado si la variable objetivo toma valores pequeños.

### 5.5 Comparación contra modelos baseline

El modelo de regresión sólo será útil si mejora referencias simples.

Se comparará contra:

```text
baseline_media_historica
baseline_mismo_mes_año_anterior
baseline_media_movil_3m
```

La pregunta no será sólo si el modelo tiene buen ajuste, sino si mejora una regla sencilla de predicción.

### 5.6 Interpretación del modelo

Una vez seleccionado el modelo final, se interpretarán los coeficientes.

Por ejemplo:

- cómo se relaciona el tamaño de la población expuesta con las nuevas maternidades equivalentes,
- si la estabilidad contractual aporta señal,
- si la proxy de hijo menor mejora la predicción,
- qué desfase de `RE_ponderado` resulta más útil,
- cuánto error típico cabe esperar en las predicciones mensuales.

---

## 6. Despliegue y monitorización

El proyecto incluirá una aplicación desarrollada con Streamlit.

Archivo principal:

```text
app/app.py
```

### 6.1 Objetivo de la aplicación

La app permitirá visualizar de forma interactiva el funcionamiento del modelo.

Podrá incluir:

- carga del dataset,
- visualización de la serie histórica,
- selección de variables explicativas,
- comparación entre predicción y valor real,
- introducción manual de escenarios,
- predicción de nuevas maternidades equivalentes para meses futuros.

### 6.2 Funcionalidad inicial

La primera versión de la aplicación puede incluir:

1. Un panel con la evolución histórica de `mat_eq_nuevas_mes`.
2. Un gráfico de correlaciones.
3. Una sección de predicción donde el usuario introduzca valores de:

```text
ppef_mujeres_25_40
pct_indef_25_40
pct_amb_fill_25_40
RE_ponderado_lag2
mes
```

4. Una salida con la predicción mensual esperada.
5. Una comparación entre modelo y baseline.

### 6.3 Monitorización futura

Un modelo de predicción no debería considerarse definitivo.

Será necesario:

- actualizar el dataset mensualmente,
- reentrenar el modelo de forma periódica,
- comparar error previsto y error real,
- detectar cambios en la relación entre variables,
- revisar si los coeficientes se mantienen estables,
- comprobar si el modelo sigue mejorando los baselines.

### 6.4 Posibles mejoras

Futuras ampliaciones podrían incluir:

- predicción del arrastre de maternidades activas en meses posteriores,
- intervalos de predicción,
- validación rolling-origin,
- comparación con Ridge o Lasso,
- generación automática de informes,
- integración con actualización mensual de datos,
- versión sintética del dataset para publicación abierta.

---

## 7. Estructura del repositorio

La estructura propuesta del repositorio es:

```text
maternity-leave-forecasting-mlr/
│
├── README.md
├── README_ESP.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── df_noves_maternitats.csv
│
├── notebooks/
│   ├── 01_data_cleaning_eda.ipynb
│   └── 02_modeling.ipynb
│
├── app/
│   └── app.py
│
└── outputs/
    ├── figures/
    ├── models/
    └── reports/
```

### `README.md`

Readme principal en inglés.

### `README_ESP.md`

Readme en español con explicación completa del proyecto.

### `requirements.txt`

Lista de librerías necesarias para ejecutar notebooks y aplicación.

Posibles dependencias:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
statsmodels
streamlit
joblib
openpyxl
```

### `data/`

Carpeta donde se guarda el dataset en formato CSV.

Si el repositorio es público, el archivo deberá ser anonimizado, agregado o sintético.

### `notebooks/01_data_cleaning_eda.ipynb`

Notebook de carga, limpieza, transformación y análisis exploratorio.

### `notebooks/02_modeling.ipynb`

Notebook de entrenamiento, evaluación e interpretación del modelo de regresión lineal múltiple.

### `app/app.py`

Aplicación Streamlit para visualizar el modelo y probar escenarios.

### `outputs/`

Carpeta para guardar gráficos, modelos entrenados y resultados.

---

## Conclusión

Este proyecto muestra cómo una técnica sencilla e interpretable como la **regresión lineal múltiple** puede aplicarse a un problema real de gestión hospitalaria.

El objetivo no es eliminar la incertidumbre, sino reducirla y hacerla más explícita. La predicción mensual de nuevas maternidades equivalentes puede ayudar a anticipar mejor necesidades de personal, planificar escenarios y convertir información administrativa dispersa en una herramienta práctica de apoyo a la decisión.

