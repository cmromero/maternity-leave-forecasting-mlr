
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# Streamlit configuration
# ============================================================

st.set_page_config(
    page_title="Maternity Leave Forecasting",
    page_icon="📈",
    layout="wide"
)

st.title("Monthly Equivalent Maternity Leave Forecasting")
st.caption(
    "Multiple Linear Regression project using fictitious hospital workforce data."
)


# ============================================================
# Helper functions
# ============================================================

@st.cache_data
def load_data():
    """
    Load the modeling dataset.

    Expected file:
    data/df_model_full.csv

    If the app is executed from the repository root:
        streamlit run app/app.py

    this relative path should work.
    """

    possible_paths = [
        Path("data/df_model_full.csv"),
        Path("../data/df_model_full.csv"),
        Path("df_model_full.csv")
    ]

    data_path = None

    for path in possible_paths:
        if path.exists():
            data_path = path
            break

    if data_path is None:
        st.error(
            "Dataset not found. Expected file: data/df_model_full.csv"
        )
        st.stop()

    df = pd.read_csv(data_path)

    df["MES"] = pd.to_datetime(df["MES"])
    df = df.sort_values("MES").reset_index(drop=True)

    return df


def smape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0

    return np.mean(
        np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]
    ) * 100


def regression_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred)
    }


@st.cache_resource
def train_model(df, features, target="mat_eq_nuevas_mes", test_size=12):
    """
    Train the selected sklearn linear regression model and calculate baselines.
    """

    # Avoid artificial zeros in initial lag variables
    max_lag = 12
    df_eval = df.iloc[max_lag:].copy()

    train = df_eval.iloc[:-test_size].copy()
    test = df_eval.iloc[-test_size:].copy()

    X_train = train[features]
    y_train = train[target]

    X_test = test[features]
    y_test = test[target]

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression())
        ]
    )

    model.fit(X_train, y_train)

    test_results = test[["MES", target, "mes"]].copy()
    test_results["pred_rlm"] = model.predict(X_test)

    # Baseline: historical global mean
    baseline_mean_value = y_train.mean()
    test_results["pred_baseline_media_global"] = baseline_mean_value

    # Baseline: historical monthly mean
    monthly_means = train.groupby("mes")[target].mean()
    test_results["pred_baseline_media_mes"] = test["mes"].map(monthly_means)

    # Metrics
    metrics_rlm = regression_metrics(
        y_test,
        test_results["pred_rlm"]
    )

    metrics_baseline_global = regression_metrics(
        y_test,
        test_results["pred_baseline_media_global"]
    )

    metrics_baseline_month = regression_metrics(
        y_test,
        test_results["pred_baseline_media_mes"]
    )

    comparison = pd.DataFrame(
        [
            {"model": "RLM model", **metrics_rlm},
            {"model": "Baseline global mean", **metrics_baseline_global},
            {"model": "Baseline monthly mean", **metrics_baseline_month},
        ]
    ).sort_values("MAE").reset_index(drop=True)

    return model, train, test, test_results, comparison


def plot_time_series(df, target):
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        df["MES"].to_numpy(),
        df[target].to_numpy(),
        marker="o",
        linewidth=1.8
    )

    ax.set_title("Historical evolution of monthly equivalent new maternity leaves")
    ax.set_xlabel("Month")
    ax.set_ylabel(target)

    return fig


def plot_correlation_heatmap(df, variables):
    corr = df[variables].corr(method="pearson")

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax
    )

    ax.set_title("Pearson correlation matrix")

    return fig


def plot_prediction_comparison(test_results, target):
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        test_results["MES"].to_numpy(),
        test_results[target].to_numpy(),
        marker="o",
        linewidth=2,
        label="Actual value"
    )

    ax.plot(
        test_results["MES"].to_numpy(),
        test_results["pred_rlm"].to_numpy(),
        marker="x",
        linestyle="--",
        linewidth=2,
        label="RLM model"
    )

    ax.plot(
        test_results["MES"].to_numpy(),
        test_results["pred_baseline_media_global"].to_numpy(),
        linestyle=":",
        label="Baseline global mean"
    )

    ax.plot(
        test_results["MES"].to_numpy(),
        test_results["pred_baseline_media_mes"].to_numpy(),
        linestyle="-.",
        label="Baseline monthly mean"
    )

    ax.set_title("Model vs baselines on test period")
    ax.set_xlabel("Month")
    ax.set_ylabel(target)
    ax.legend()

    return fig


# ============================================================
# Load data and train model
# ============================================================

df = load_data()

target = "mat_eq_nuevas_mes"

# Final model features according to project README / Streamlit initial version
model_features = [
    "ppef_mujeres_25_40",
    "RE_ponderado_lag1",
    "mes"
]

required_cols = ["MES", target] + model_features

missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    st.error(f"Missing columns in dataset: {missing_cols}")
    st.stop()

model, train, test, test_results, comparison = train_model(
    df=df,
    features=model_features,
    target=target,
    test_size=12
)


# ============================================================
# Sidebar
# ============================================================

st.sidebar.header("Model inputs")

st.sidebar.write(
    "Use the controls below to simulate a future monthly scenario."
)

ppef_default = float(df["ppef_mujeres_25_40"].iloc[-1])
re_default = float(df["RE_ponderado_lag1"].iloc[-1])
mes_default = int(df["mes"].iloc[-1])

ppef_input = st.sidebar.number_input(
    "ppef_mujeres_25_40",
    min_value=0.0,
    value=round(ppef_default, 2),
    step=10.0,
    help="Equivalent workforce of women aged 25–40."
)

re_input = st.sidebar.number_input(
    "RE_ponderado_lag1",
    min_value=0.0,
    value=round(re_default, 2),
    step=1.0,
    help="Weighted active pregnancy-risk indicator lagged by one month."
)

mes_input = st.sidebar.slider(
    "mes",
    min_value=1,
    max_value=12,
    value=mes_default,
    help="Month number used as a seasonal variable."
)


# ============================================================
# Main layout
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Historical series",
        "Correlation analysis",
        "Model vs baseline",
        "Scenario prediction"
    ]
)


# ------------------------------------------------------------
# Tab 1: historical evolution
# ------------------------------------------------------------

with tab1:
    st.subheader("Historical evolution")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Monthly average",
        f"{df[target].mean():.2f}"
    )

    col2.metric(
        "Minimum",
        f"{df[target].min():.2f}"
    )

    col3.metric(
        "Maximum",
        f"{df[target].max():.2f}"
    )

    fig = plot_time_series(df, target)
    st.pyplot(fig)

    with st.expander("View dataset"):
        st.dataframe(df)


# ------------------------------------------------------------
# Tab 2: correlation analysis
# ------------------------------------------------------------

with tab2:
    st.subheader("Correlation analysis")

    correlation_variables = [
        target,
        "ppef_mujeres_25_40",
        "RE_ponderado_lag1",
        "mes"
    ]

    fig = plot_correlation_heatmap(df, correlation_variables)
    st.pyplot(fig)

    st.write(
        """
        This heatmap shows the Pearson correlation between the target variable
        and the explanatory variables used in the initial deployed model.
        Correlation does not imply causality, but it helps understand whether
        the predictors contain useful signal.
        """
    )


# ------------------------------------------------------------
# Tab 3: model vs baselines
# ------------------------------------------------------------

with tab3:
    st.subheader("Model vs baseline comparison")

    st.dataframe(comparison)

    fig = plot_prediction_comparison(test_results, target)
    st.pyplot(fig)

    best_model_mae = comparison.loc[
        comparison["model"] == "RLM model",
        "MAE"
    ].iloc[0]

    best_baseline = comparison[
        comparison["model"] != "RLM model"
    ].sort_values("MAE").iloc[0]

    improvement = (
        (best_baseline["MAE"] - best_model_mae)
        / best_baseline["MAE"]
        * 100
    )

    st.info(
        f"The RLM model MAE is {best_model_mae:.2f}. "
        f"The best baseline is '{best_baseline['model']}' "
        f"with MAE {best_baseline['MAE']:.2f}. "
        f"The estimated improvement is {improvement:.1f}%."
    )


# ------------------------------------------------------------
# Tab 4: scenario prediction
# ------------------------------------------------------------

with tab4:
    st.subheader("Manual scenario prediction")

    scenario_df = pd.DataFrame(
        {
            "ppef_mujeres_25_40": [ppef_input],
            "RE_ponderado_lag1": [re_input],
            "mes": [mes_input]
        }
    )

    prediction = model.predict(scenario_df)[0]

    st.metric(
        "Expected monthly equivalent new maternity leaves",
        f"{prediction:.2f}"
    )

    st.write("Scenario inputs:")
    st.dataframe(scenario_df)

    baseline_month_value = train.groupby("mes")[target].mean().get(
        mes_input,
        train[target].mean()
    )

    st.write(
        f"For comparison, the historical monthly-mean baseline for month "
        f"{mes_input} is **{baseline_month_value:.2f}**."
    )

    st.write(
        """
        This prediction should be interpreted as an expected aggregate value,
        not as an individual-level forecast. The model reduces uncertainty,
        but it does not eliminate it.
        """
    )
