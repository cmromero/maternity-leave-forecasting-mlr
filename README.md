# Monthly forecasting of equivalent new maternity leaves in a hospital

## Educational multiple linear regression project applied to workforce planning

This repository develops a complete **quantitative forecasting project applied to hospital human resources management**.

The objective is to build, evaluate, and deploy a model capable of estimating the expected monthly volume of **equivalent new maternity leaves**. The idea is not to predict individual decisions, but to anticipate an aggregate phenomenon that may affect replacement planning, healthcare organization, and budget forecasting.

The project follows a complete workflow:

1. dataset construction and cleaning;
2. exploratory analysis;
3. creation of predictive variables;
4. training of multiple linear regression models;
5. comparison against simple baselines;
6. deployment of an interactive app with Streamlit.

> **Note about the data:** the data included in this repository are fictitious and are used only for educational and illustrative purposes. They do not correspond to real data from any institution and do not contain personal or identifiable administrative information.

---

## 1. Forecasting question

The main question of the project is:

> Can we anticipate the monthly volume of equivalent new maternity leaves among female workers in a hospital using aggregated information on workforce size, contractual stability, approximate family structure, and pregnancy-risk leave?

The forecast is made at a **monthly** scale, because this frequency is useful for operational and budget planning.

---

## 2. Target variable

The target variable is:

```text
mat_eq_nuevas_mes
```

It represents **monthly equivalent new maternity leaves**.

It is calculated as:

```text
mat_eq_nuevas_mes = dias_mat_nuevas_mes / dias_naturales_mes
```

The logic is simple: a maternity leave that starts at the beginning of the month generates more monthly workload than one that starts at the end.

For example, if a maternity leave starts on day 16 of a 30-day month, it contributes 15 days within that month:

```text
15 / 30 = 0.5
```

That is why this variable is more informative than a simple count of new maternity leaves. It better measures the equivalent monthly impact of the phenomenon.

---

## 3. Explanatory variables

The dataset uses aggregated monthly variables. The main ones are:

| Variable | Description |
|---|---|
| `MES` | Reference month. |
| `mat_eq_nuevas_mes` | Target variable: monthly equivalent new maternity leaves. |
| `ppef_mujeres_25_40` | Equivalent workforce of women aged 25 to 40. It approximates the size of the exposed population. |
| `pct_indef_25_40` | Percentage of women aged 25 to 40 with a permanent contract. It approximates labor stability. |
| `pct_amb_fill_25_40` | Percentage of women aged 25 to 40 with at least one minor child declared in payroll. It is an administrative proxy for family structure. |
| `RE_ponderado_lag1` | Weighted pregnancy-risk leave indicator with a one-month lag. |
| `RE_ponderado_lag2` | Weighted pregnancy-risk leave indicator with a two-month lag. |
| `RE_ponderado_lag3` | Weighted pregnancy-risk leave indicator with a three-month lag. |
| `mat_eq_lag1` | Value of the target variable in the previous month. |
| `mat_eq_lag12` | Value of the target variable in the same month of the previous year. |
| `mes` | Month number, used as a seasonality variable. |

The pregnancy-risk leave variable is considered especially relevant because it may work as an **early signal**: some pregnancy-risk leave situations later end in maternity leave.

---

## 4. Project workflow

The project is organized into three main notebooks.

### 4.1 Notebook 01 — Data Cleaning & EDA

File:

```text
notebooks/01_data_cleaning_eda.ipynb
```

This notebook prepares the data and performs the exploratory analysis.

It includes:

- loading the initial file;
- converting `MES` to date format;
- chronological sorting;
- reviewing types, missing values, and descriptive statistics;
- visualizing the evolution of the target variable;
- visual comparison with `ppef_mujeres_25_40` and `RE_ponderado`;
- creation of lagged variables;
- Pearson and Spearman correlation analysis;
- collinearity analysis using a correlation matrix and VIF;
- export of the clean dataset for modeling.

The main output of this notebook is:

```text
data/df_model_full.csv
```

This file contains all clean candidate variables prepared for modeling.

---

### 4.2 Notebook 02 — Modeling

File:

```text
notebooks/02_modeling.ipynb
```

This notebook builds and compares **multiple linear regression** models.

The process includes:

- loading `df_model_full.csv`;
- removing the first months without enough history for the lagged variables;
- temporal train/test split;
- controlled generation of variable combinations;
- training models with `scikit-learn LinearRegression`;
- training models with `statsmodels OLS`;
- model comparison using MAE, RMSE, R², and sMAPE;
- careful selection of the best model;
- analysis of predictions, residuals, coefficients, and collinearity of the final model.

The model search does not consist of testing any combination without criteria. The size of the models is limited and redundant combinations are avoided, such as including several very similar lags of the same indicator.

---

### 4.3 Notebook 03 — Baselines and Model Comparison

File:

```text
notebooks/03_baselines_and_model_comparison.ipynb
```

This notebook answers the key question:

> Does the multiple linear regression model really improve simple prediction rules?

To do this, it compares the best MLR model against several baselines:

| Baseline | Description |
|---|---|
| Global historical mean | Always predicts the historical average of the training set. |
| Historical mean by month | Predicts the historical average of that same month of the year. |
| Last observed value | Uses the previous month’s value as the prediction. |
| Same month of the previous year | Uses the value observed 12 months earlier as the prediction. |
| 3-month moving average | Uses the average of the previous three months. |

This comparison is essential. A predictive model only makes sense if it improves simple, transparent, and easy-to-implement alternatives.

---

## 5. Evaluation criterion

The main metric of the project is:

```text
MAE — Mean Absolute Error
```

MAE is interpreted in the same units as the target variable. For example, an MAE of 1.2 means that the model is wrong, on average, by about 1.2 monthly equivalent maternity leaves.

The following are also calculated:

```text
RMSE
R²
sMAPE
```

Model selection is not based only on the lowest error. The following are also taken into account:

- parsimony;
- interpretability;
- stability;
- absence of severe collinearity;
- operational usefulness;
- improvement over baselines.

---

## 6. Streamlit application

File:

```text
app/app.py
```

The project includes an interactive application developed with **Streamlit**.

The app allows users to:

- load the modeling dataset;
- visualize the historical series of `mat_eq_nuevas_mes`;
- consult basic descriptive metrics;
- view a correlation chart;
- compare the model against baselines;
- enter manual scenarios;
- obtain an expected monthly prediction.

The application uses a deployable version of the model based on:

```text
StandardScaler + LinearRegression
```

and allows users to simulate scenarios by entering values for:

```text
ppef_mujeres_25_40
RE_ponderado_lag1
mes
```

---

## 7. How to run the project

### 7.1 Clone the repository

```bash
git clone https://github.com/your_username/maternity-leave-forecasting-mlr.git
cd maternity-leave-forecasting-mlr
```

### 7.2 Create the environment and install dependencies

```bash
pip install -r requirements.txt
```

### 7.3 Run the notebooks

Recommended order:

```text
notebooks/01_data_cleaning_eda.ipynb
notebooks/02_modeling.ipynb
notebooks/03_baselines_and_model_comparison.ipynb
```

### 7.4 Reproduce and run the Streamlit application

The app is located at:

```text
app/app.py
```

To reproduce it locally, the repository must keep this structure:

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

The app expects the modeling dataset to be available at:

```text
data/df_model_full.csv
```

If this file does not exist yet, run the notebooks first in the recommended order:

```text
notebooks/01_data_cleaning_eda.ipynb
notebooks/02_modeling.ipynb
notebooks/03_baselines_and_model_comparison.ipynb
```

Then, from the root of the repository, install the dependencies:

```bash
pip install -r requirements.txt
```

Finally, launch the app with:

```bash
streamlit run app/app.py
```

The application will normally open automatically in the browser. If it does not, open:

```text
http://localhost:8501
```

To stop the app and return to the terminal, press:

```text
Ctrl + C
```

#### What the app does

The Streamlit app reproduces the deployable part of the project. It:

- loads `data/df_model_full.csv`;
- converts `MES` to datetime format;
- trains a deployable regression pipeline using `StandardScaler + LinearRegression`;
- compares the model against simple baselines;
- displays the historical series of `mat_eq_nuevas_mes`;
- shows a Pearson correlation heatmap;
- allows manual scenario simulation using:

```text
ppef_mujeres_25_40
RE_ponderado_lag1
mes
```

The output is an expected monthly value of equivalent new maternity leaves.

#### Troubleshooting

If Streamlit cannot find the dataset, check that the file exists here:

```text
data/df_model_full.csv
```

and that the command is being executed from the repository root, not from inside the `app/` folder.

If the required packages are missing, reinstall the dependencies:

```bash
pip install -r requirements.txt
```

---

## 8. Repository structure

```text
maternity-leave-forecasting-mlr/
│
├── README.md
├── README_ESP.md (Spanish readme)
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

### Main files

| File | Function |
|---|---|
| `data/dataframe.xlsx` | Initial fictitious dataset. |
| `data/df_model_full.csv` | Clean dataset used for modeling. |
| `notebooks/01_data_cleaning_eda.ipynb` | Cleaning, transformation, and EDA. |
| `notebooks/02_modeling.ipynb` | MLR model construction and selection. |
| `notebooks/03_baselines_and_model_comparison.ipynb` | Comparison against baselines and final evaluation. |
| `app/app.py` | Interactive Streamlit application. |
| `requirements.txt` | Project dependencies. |
| `.gitignore` | Exclusions for local files, checkpoints, and unnecessary outputs. |

---

## 9. Expected results

This project does not aim to eliminate uncertainty. It aims to reduce it in a measurable way.

A model will be considered useful if it:

- clearly improves simple baselines;
- maintains an acceptable mean error;
- uses few variables;
- produces interpretable results;
- can be integrated into a planning support tool.

In the final comparison, if the multiple linear regression model reduces MAE compared with the best baseline, it can be considered a first valid tool to support monthly equivalent new maternity leave forecasting.

---

## 10. Limitations

The project has important limitations:

- it works with aggregated monthly data;
- it does not predict individual decisions;
- the variable `pct_amb_fill_25_40` is an imperfect administrative proxy;
- the relationship between pregnancy-risk leave and maternity leave may change over time;
- the results depend on the available historical period;
- the model must be reviewed and retrained periodically.

In addition, the published dataset is fictitious, so the objective of the repository is methodological and educational.

---

## 11. Next improvements

Possible extensions of the project:

- rolling-origin validation;
- comparison with Ridge and Lasso;
- prediction intervals;
- improvement of the `RE_ponderado` indicator;
- modeling the carry-over effect of active maternity leaves;
- automated monthly updating;
- report generation;
- deployment of the app on Streamlit Community Cloud.

---

## Conclusion

This project shows how a simple and interpretable technique such as **multiple linear regression** can be applied to a realistic hospital management problem.

Monthly forecasting of equivalent new maternity leaves does not eliminate uncertainty, but it helps structure it more clearly. It turns scattered administrative information into a useful estimate for planning, scenario comparison, and better decision-making.

In this sense, the value of the project is not only in the final model, but in the process: properly defining the target variable, building the dataset, comparing alternatives, measuring error, and always checking the result against simple baselines.
