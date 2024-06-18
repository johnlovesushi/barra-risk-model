sys.path.append("../../../../forecastos")

# Now you can import the library as usual
import forecastos as fos

fos.api_endpoint = os.environ.get("FH_API_ENDPOINT")
fos.fh_api_key = os.environ.get("FORECASTOS_API_KEY")

import pandas as pd
import numpy as np
import os, sys

# Now you can import the library as usual
import investos as inv

#### 4.1 Loadings

# 4.1 Idiosync only for now (gonna use off-the-shelf PCA based for this)

# Get actual returns (for risk model)
df_actual_returns = (
    fos_h.find_feature("return_prev_1d_open", "trading", "return")
    .get_df()
    .sort_values(["id", "datetime"])
)
df_actual_returns = df_actual_returns[
    (df_actual_returns.datetime <= max_dt) & (df_actual_returns.id.isin(company_ids))
]
# df_actual_returns = df_actual_returns.pivot(index='datetime', columns='id', values='value')
df_actual_returns = df_actual_returns.fillna(0)

# Join and calc industry loadings
df_loadings = df_actual_returns.merge(
    fos_h.find_feature("rbics_industry", "corporation", "classification")
    .get_df()
    .rename(columns={"value": "industry"}),
    on=["id"],
    how="left",
)

df_loadings = pd.get_dummies(df_loadings, columns=["industry"])
for col in df_loadings.columns:
    if col.startswith("industry_"):
        df_loadings[col] = df_loadings[col].astype(int)

df_loadings = df_loadings.rename(columns={"value": "return_1d"})

#### 4.2 Calc Factor Returns

from sklearn.linear_model import LinearRegression

# Dictionary to store regression results
factor_returns = {}

# Iterating over each distinct date
for d in df_loadings["datetime"].unique():
    # Isolating the data for the current date
    df_current = df_loadings[df_loadings["datetime"] == d]

    # Defining the dependent variable (y) and independent variables (X)
    y = df_current["return_1d"]
    X = df_current.drop(columns=["return_1d", "datetime", "id"])

    # Performing linear regression
    model = LinearRegression(fit_intercept=False).fit(X, y)

    # Storing the coefficients and intercept
    factor_returns[d] = {
        "coefficients": model.coef_,
        "intercept": model.intercept_,
        "feature_names": model.feature_names_in_,
        "r2": model.score(X, y),
    }

list_to_insert = [
    [k, factor_returns[k]["r2"], *factor_returns[k]["coefficients"]]
    for k in factor_returns
]
cols = df_loadings.drop(columns=["return_1d", "datetime", "id"]).columns

# df_factor_returns = df.append(pd.Series(list_to_insert, index=['date', 'r2', *cols]), ignore_index=True)  # using append
df_factor_returns = pd.DataFrame(list_to_insert, columns=["datetime", "r2", *cols])

#### 4.3 Calc idiosyncratic risk

factor_cols = [
    col
    for col in df_factor_returns.columns
    if col in df_loadings.columns and col != "datetime"
]

# Merge the DataFrames on 'date'
df_idio = pd.merge(
    df_loadings, df_factor_returns, on="datetime", suffixes=("", "_factor_returns")
)

# Multiplying matching columns
for col in factor_cols:
    df_idio[f"{col}_calc_f_r"] = df_idio[col] * df_idio[f"{col}_factor_returns"]

# # Dropping the extra columns
df_idio = df_idio.drop(columns=factor_cols)
df_idio = df_idio.drop(columns=[f"{col}_factor_returns" for col in factor_cols])

# Summing the specified columns
df_idio["factor_return_1d"] = df_idio[[f"{col}_calc_f_r" for col in factor_cols]].sum(
    axis=1
)
df_idio["factor_return_1d_error"] = df_idio["factor_return_1d"] - df_idio["return_1d"]

df_idio = df_idio[
    ["datetime", "id", "return_1d", "factor_return_1d", "factor_return_1d_error"]
]

# Calculate idiosync_return_var from return error
df_idio["idio_risk_252d"] = (
    df_idio.sort_values(by=["id", "datetime"])
    .groupby("id")["factor_return_1d_error"]
    .rolling(window=252)
    .var()
    .reset_index(level=0, drop=True)
)
df_idio = df_idio.set_index("datetime").pivot(columns="id", values="idio_risk_252d")

#### 4.4 Calculate CoVar Factor Matrix

# Defining the rolling window size
window_size = 252


# Function to calculate the covariance matrix for each rolling window
def rolling_covariance(df, window):
    return df.rolling(window=window).cov()


# Applying the function to the dataframe
df_factor_covar = rolling_covariance(
    df_factor_returns.set_index("datetime").drop(columns=["r2"]), window_size
).dropna()

#### 4.5 Cleanup

df_loadings = (
    df_loadings.drop(columns=["return_1d"])
    .melt(id_vars=["datetime", "id"], var_name="factor", ignore_index=True)
    .set_index(["datetime", "factor"])
    .pivot(columns="id")
    .droplevel(0, axis=1)
    .fillna(0)
)

df_loadings = df_loadings.sort_index().sort_index(axis=1)
df_factor_covar = df_factor_covar.sort_index().sort_index(axis=1)

df_idio = df_idio.fillna(0)
df_factor_covar = df_factor_covar.fillna(0)

#### 4.6 Custom Risk Model

from investos.portfolio.risk_model import BaseRisk


class FactorRiskTest(BaseRisk):
    """Multi-factor risk model."""

    def __init__(
        self, factor_covariance, factor_loadings, idiosyncratic_variance, **kwargs
    ):
        super().__init__(**kwargs)

        self.factor_covariance = factor_covariance
        self.factor_loadings = factor_loadings
        self.idiosyncratic_variance = idiosyncratic_variance

        self._drop_excluded_assets()

    def _estimated_cost_for_optimization(self, t, w_plus, z, value):
        """Optimization (non-cash) cost penalty for assuming associated asset risk.

        Used by optimization strategy to determine trades.

        Not used to calculate simulated costs for backtest performance.
        """
        factor_covar = util.values_in_time(
            self.factor_covariance, t, lookback_for_closest=True
        )
        factor_load = util.values_in_time(
            self.factor_loadings, t, lookback_for_closest=True
        )
        idiosync_var = util.values_in_time(
            self.idiosyncratic_variance, t, lookback_for_closest=True
        )

        self.expression = cvx.sum_squares(cvx.multiply(np.sqrt(idiosync_var), w_plus))

        risk_from_factors = factor_load.T @ factor_covar @ factor_load

        self.expression += w_plus @ risk_from_factors @ w_plus.T

        print(self.expression)

        return self.expression, []

    def _drop_excluded_assets(self):
        self.factor_loadings = self._remove_excl_columns(self.factor_loadings)
        self.idiosyncratic_variance = self._remove_excl_columns(
            self.idiosyncratic_variance
        )

    def _remove_excl_columns(self, pd_obj):
        return util.remove_excluded_columns_pd(
            pd_obj,
            exclude_assets=self.exclude_assets,
        )


