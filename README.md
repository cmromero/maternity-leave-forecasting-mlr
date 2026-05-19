# Sick Leave Absenteeism → Temporary Coverage Contracts  
## Simple linear regression project to estimate the effect of work-related absenteeism on temporary hiring

This repository presents a complete data analysis project focused on answering a very specific question in the field of human resources:

**How many temporary coverage contracts are generated when work-related absenteeism due to temporary incapacity (IT) increases?**

The central idea is to build a **linear regression** model using historical monthly data to estimate the relationship between both variables and transform that relationship into a simple decision-support tool.

Rather than presenting only a final model, this project reproduces the full workflow of a quantitative prediction project: from problem formulation to deployment in a small web application.

---

## 1. Objective definition

Every predictive project begins with a well-defined question. In this case, the objective is to estimate the impact of increasing absenteeism due to temporary incapacity (IT) on the need to generate temporary coverage contracts.

### Project question

> How does the number of temporary coverage contracts vary when the monthly volume of IT absenteeism increases?

### Target variable

- **y = temporary_coverage_contracts**

### Main explanatory variable

- **X = IT_absenteeism**
- In the final application, this variable may be entered as the **number of new IT cases** or as the equivalent monthly indicator, depending on the final dataset definition.

### Analysis horizon

- Historical data with **monthly** frequency
- Model designed for **quantitative estimation** and scenario simulation

### Practical use

This model can serve as a decision-support tool to:

- anticipate temporary hiring needs,
- improve human resources planning,
- justify organizational decisions with a quantitative basis,
- explore scenarios of increasing or decreasing absenteeism.

---

## 2. Data acquisition

The next step is to gather and document the information needed to build the model.

This project works with a CSV file containing historical monthly observations of the two main variables:

- IT absenteeism,
- temporary coverage contracts.

### Dataset

Main file:

- `data/absenteeism_contracts.csv`

### Expected structure

Each row represents one month in the historical series. The dataset will include, at minimum, the following columns:

- `date`
- `it_absenteeism_days`
- `temporary_coverage_contract_days`

Conceptual example:

| date      | it_absenteeism_days | temporary_coverage_contract_days |
|-----------|--------------------:|---------------------------------:|
| 2021-01   | 125                 | 18                               |
| 2021-02   | 138                 | 21                               |
| 2021-03   | 149                 | 24                               |

### Note on data origin

The data included in `absenteeism_contracts.csv` is fictitious.

It has been created for educational and illustrative purposes only. It does not correspond to real institutional or administrative data, although it is structured to resemble a realistic monthly workforce planning dataset.

---

## 3. Exploratory analysis

Before training any model, it is necessary to observe the data and understand what pattern it contains.

In this project, the exploratory analysis is developed in the notebook:

- `notebooks/01_data_cleaning_eda.ipynb`

### What is analyzed in this phase

- general structure of the dataset,
- data types and missing values,
- possible errors or atypical records,
- distribution of the variables,
- monthly time evolution,
- relationship between IT absenteeism and coverage contracts,
- linear correlation between both variables.

### Objective of this phase

The purpose of the EDA is to check whether there is a sufficiently clear and stable relationship to justify using a **simple linear regression** model.

In particular, this notebook should end by showing that:

- there is a positive association between both variables,
- the scatter plot suggests an approximately linear trend,
- the problem can be formulated as an interpretable linear estimation problem.

---

## 4. Dataset preparation

Models only work well when the data is properly defined and organized. For this reason, a cleaning and preparation phase is carried out before modeling.

This phase is also included in:

- `notebooks/01_data_cleaning_eda.ipynb`

### Planned preparation tasks

- parsing and sorting the date variable,
- checking for duplicates,
- handling missing values,
- validating numeric data types,
- standardizing variable names,
- selecting the final columns needed for modeling.

### Final dataset for modeling

The result of this phase is a clean, consistent dataset ready to build the regression model.

---

## 5. Model construction

The modeling phase is developed in the notebook:

- `notebooks/02_modeling.ipynb`

Here, **two linear regression approaches** are trained and compared, using a test set to select the most useful model.

### Models to compare

#### Model 1. Linear regression with `scikit-learn`
A model focused on practical prediction, simple training, and easy integration into a later application.

#### Model 2. Linear regression with `statsmodels` (OLS)
A model focused on statistical interpretation, useful for analyzing coefficients, significance, and overall fit.

### Evaluation strategy

The notebook will split the historical data into:

- **train**
- **test**

And compare both approaches using metrics such as:

- MAE
- RMSE
- R²

### Expected result

Select the model that offers the best balance between:

- predictive capacity,
- interpretability,
- ease of deployment.

### Model interpretation

One of the main interests of this project is not only to predict, but also to interpret the regression coefficient.

That is:

> estimate how many additional temporary contracts are generated, on average, for each unit increase in IT absenteeism.

---

## 6. Deployment and monitoring

A model gains real value when it is put into use. For this reason, this repository includes a small web application developed with Streamlit to show the model in action.

Main file:

- `app/app.py`

### Application functionality

The app will allow the user to enter as input:

- a number of new IT cases, or
- the monthly absenteeism indicator defined in the project,

and will return as output:

- the estimated number of expected temporary coverage contracts.

### Purpose of the app

To transform the model into a simple, understandable, and reusable tool designed for scenario simulation.

### Future monitoring

Although this repository shows a first functional version, any real use of the model would require:

- periodically updating the dataset,
- reassessing model fit,
- checking whether the relationship between variables remains stable over time.

---

## Repository structure

```text
sick-leave-temporary-contracts-regression/
│
├── README.md
├── README_ESP.md (readme in Spanish)
├── requirements.txt
├── .gitignore
│
├── data/
│   └── absenteeism_contracts.csv
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
    └── model/
```
