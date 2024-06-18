sys.path.append("../../../../forecastos")

# Now you can import the library as usual
# import forecastos as fos
#
# fos.api_endpoint = os.environ.get("FH_API_ENDPOINT")
# fos.fh_api_key = os.environ.get("FORECASTOS_API_KEY")

import pandas as pd
import numpy as np
import os, sys

# Now you can import the library as usual
import investos as inv
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression
#### 4.1 Loadings

# 4.1 Idiosync only for now (gonna use off-the-shelf PCA based for this)

# Get actual returns (for risk model)
# df_actual_returns = (
#     fos_h.find_feature("return_prev_1d_open", "trading", "return")
#     .get_df()
#     .sort_values(["id", "datetime"])
# )
# df_actual_returns = df_actual_returns[
#     (df_actual_returns.datetime <= max_dt) & (df_actual_returns.id.isin(company_ids))
# ]
# # df_actual_returns = df_actual_returns.pivot(index='datetime', columns='id', values='value')
# df_actual_returns = df_actual_returns.fillna(0)
#
# # Join and calc industry loadings
# df_loadings = df_actual_returns.merge(
#     fos_h.find_feature("rbics_industry", "corporation", "classification")
#     .get_df()
#     .rename(columns={"value": "industry"}),
#     on=["id"],
#     how="left",
# )

# Step 1: loading the dataframe and join them together
# loading
df_loadings = pd.read_parquet('john_rbics_industry.parquet')
df_actual_returns = pd.read_parquet('john_return_prev_1d_open.parquet').fillna(0).rename(columns={"value": "return_1d"})
df_sizes = pd.read_parquet('john_market_cap_open_dil.parquet').fillna(0).rename(columns={"value": "size"})
# log and then winsorize
#df_sizes['size'] = np.log(df_sizes['size'])
df_values = pd.read_parquet('john_market_cap_open_dil_to_net_income_ltm.parquet').fillna(0).rename(columns={"value": "value"})
df_loadings = pd.get_dummies(df_loadings, columns=["industry"])

# merge these dataframe into one
df_sizes = df_sizes.merge(df_values, how='left', on=['id','datetime'])
df_actual_returns = df_actual_returns.merge(df_sizes, how='left', on=['id','datetime'])
df_loadings = pd.merge(df_actual_returns, df_loadings, how='left', on=None, left_on='id', right_on='id', left_index=False)


#df_loadings = pd.get_dummies(df_loadings, columns=["industry"])
for col in df_loadings.columns:
    if col.startswith("industry_"):
        df_loadings[col] = df_loadings[col].astype(int)

df_loadings = df_loadings.rename(columns={"value": "return_1d"})

# Step 2: Calculate momentum (e.g., 12-month cumulative return)

# sliding window size for volatiltiy and momentum
look_back_period_volatility = 252  # Approx. 12 months for daily data
look_back_period_momentum = 30
#df_loadings['momentum'] = df_loadings.groupby('id')['return_1d'].apply(lambda x: x.rolling(window=look_back_period_momentum).apply(lambda y: np.prod(1 + y) - 1))
df_loadings['volatility'] = df_loadings.groupby('id')['return_1d'].apply(lambda x: x.rolling(window=look_back_period_volatility).std()).reset_index(level = 'id').rename(columns={"return_1d": "volatility"}).volatility
df_loadings['momentum'] = df_loadings.groupby('id')['return_1d'].rolling(window=look_back_period_momentum).apply(lambda x: np.prod(1 + x) - 1, raw = True).reset_index(level = 'id').rename(columns={"return_1d": "momentum"}).momentum
# drop those nan values less than then window size
df_loadings.dropna(inplace = True)

# Step 3: identify some inf values and replace them as 0

# generate an inf mask and count the True value
df_numeric = df_loadings.apply(pd.to_numeric, errors='coerce')
inf_mask = np.isinf(df_numeric)
inf_counts = inf_mask.sum()

print('inf value count in each of the factors')
print(inf_counts)

# replace np.inf and -np.inf as 0 in df_loadings
df_loadings.replace([np.inf, -np.inf], 0, inplace=True)

# Step 4: adding a country factor and fake a quality factor
#np.random.seed(42)  # For reproducibility
#df_loadings['country'] = 1
#df_loadings['quality'] = np.random.randn(len(df_loadings))  # Simulating quality factor

# Step 5: slice the dataframe after 2022-09-07
min_dt = '2020-09-07'
df_loadings = df_loadings[df_loadings.datetime >= min_dt]

#### 4.2 Calc Factor Returns

# based on the datetime to obtain a regression on all different stocks

# Step 1: filter the outlier and standarized the style factors
def get_filter_standarized_df(df: pd.DataFrame) -> pd.DataFrame:
    def get_median_filter(x, x_M, MAD, n=3):
        # filter extreme value. MAD method
        # MAD is better than 3std away from mean method, and Inter Quantile Range(IQR) method
        # 𝑥̃𝑖 = 𝑥𝑀 + 𝑛 ∗ 𝐷_𝑀𝐴𝐷, 𝑖𝑓 𝑥𝑖 > 𝑥𝑀 + 𝑛 ∗ 𝐷𝑀𝐴𝐷
        # 𝑥̃𝑖 = 𝑥𝑀 −𝑛 ∗ 𝐷𝑀𝐴𝐷,  𝑖𝑓 𝑥𝑖 < 𝑥𝑀 −𝑛 ∗ 𝐷𝑀𝐴𝐷
        # 𝑥̃𝑖 = xi else

        # D_MAD = abs(x - x_M)
        modified_z_score = 0.6745 * (x - x_M) / MAD
        if abs(modified_z_score) > n:
            return modified_z_score * MAD / 0.6745 + x_M
        else:
            return x

    def standardize(x, mean, std):
        # z-socre all style factors
        return (x - mean) / std

    # filter out the columns need to apply a standarization: the column should only include style factors
    style_factors_cols = [col for col in df.columns if
                          not col.startswith('industry_') and col not in ['country', 'datetime', 'id', 'return_1d']]

    # apply outlier filter
    # medians = df[style_factors_cols].median()
    for col in style_factors_cols:
        #     MAD = median(np.absolute(df.loc[:,col] - medians[col]))
        #     df.loc[:,col] = df[col].apply(lambda x: get_median_filter(x, medians[col],MAD))
        # using winsorize
        df.loc[:, col] = winsorize(df.loc[:, col], limits=[0.025, 0.025])

    # apply factor standarization
    means = df[style_factors_cols].mean()
    stds = df[style_factors_cols].std()

    for col in style_factors_cols:
        df.loc[:, col] = standardize(df[col], means[col], stds[col])

    return df

# Step 2: identify factor effectiveness
def get_t_statistics(model,X,y):
    predictions = model.predict(X)
    residuals = y - predictions
    # Calculate the residual standard error
    rss = (residuals ** 2).sum()
    n = len(y)      # number of sample
    p = X.shape[1]  # number of factors

    rse = (rss / (n - p - 1)) ** 0.5
    print(n,p,rse)
    # Calculate t-values for coefficients
    t_values = model.coef_ / rse
    print(model.coef_,rse)
    return t_values


# Step 3: Dictionary to store regression results

"""
1.
it comes out with an warning like this:
# /var/folders/sg/1nh38dx53_zc3kxcq1cq1f000000gn/T/ipykernel_29543/1917117034.py:11: RuntimeWarning: invalid value encountered in divide
#  t_values = model.coef_ / rse

this is caused by t_values = model.coef_ / rse while certain model.coef_ value is too low (about e-17 level). Overall the code is fine.
We can just mute this warning later

2. some results shows [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.] 0.0 which model.coef_ is a zero list, and rse is zero 
as well
"""

factor_returns = {}

# Iterating over each distinct date/ month/ year
# modify it to monthly
# period = 'D'
# df_loadings["datetime"] = df_loadings["datetime"].dt.to_period(period)
for d in df_loadings["datetime"].unique():
    # Isolating the data for the current date
    df_current = df_loadings[df_loadings["datetime"] == d]

    # outlier modification and factor standarization
    df_current = get_filter_standarized_df(df_current)

    # identify factor effectiveness
    # Defining the dependent variable (y) and independent variables (X)
    y = df_current["return_1d"]
    X = df_current.drop(columns=["return_1d", "datetime", "id"])

    # Performing linear regression
    model = LinearRegression(fit_intercept=False).fit(X, y)

    t_values = get_t_statistics(model, X, y)

    # Print t-values for each factor
    # print("T-Values:")
    # for i, factor in enumerate(X.columns):
    #     print(f"{factor}: {t_values[i]}")

    # Storing the coefficients and intercept
    factor_returns[d] = {
        "coefficients": model.coef_,
        "intercept": model.intercept_,
        "feature_names": model.feature_names_in_,
        "r2": model.score(X, y),
        # t_values added to the factor_returns dataframe
        "t-values": abs(t_values)
    }


#### 4.3 Calc Factor Returns

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
    [k, factor_returns[k]["r2"], *factor_returns[k]["t-values"],*factor_returns[k]["coefficients"]]
    for k in factor_returns    # iterative through the factor returns dict
]
#print(list_to_insert)
cols = df_loadings.drop(columns=["return_1d", "datetime", "id"]).columns

# update
cols_with_return = ['returns_' + col for col in cols]
cols_with_t_values = ['t_values_' + col for col in cols]

# df_factor_returns = df.append(pd.Series(list_to_insert, index=['date', 'r2', *cols]), ignore_index=True)  # using append
df_factor_returns = pd.DataFrame(list_to_insert, columns=["datetime", "r2", *cols_with_t_values, *cols_with_return])


df_factor_returns_cp = df_factor_returns.copy()
df_factor_returns_cp.set_index('datetime', inplace=True)

# Group by year and month, then calculate the mean for each group
monthly_summary = df_factor_returns_cp.resample('2YE').mean()

# Reset index to have the 'datetime' column back if needed
monthly_summary.reset_index(inplace=True)

from tabulate import tabulate

for index, row in monthly_summary.iterrows():
    print(f"date: {row['datetime']}")
    for column in monthly_summary.columns[1:]:
        print(f"{column}: {row[column]}")
    print("-" * 30)

#### 4.3 Calc idiosyncratic risk: TODO

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


