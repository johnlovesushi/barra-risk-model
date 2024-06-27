import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler
# Function to winsorize and standardize within each group
def winsorization_standardization_group(group,cols):
    for col in cols:
        # Winsorizing numeric columns at 5th and 95th percentiles within the group
        group[col] = winsorize(group[col], limits=[0.05, 0.05])

    # Ensure no infinite values are left
    group[cols] = np.nan_to_num(group[cols], nan=0.0, posinf=0.0, neginf=0.0)

    # Standardizing numeric columns within the group
    scaler = StandardScaler()
    group[cols] = scaler.fit_transform(group[cols])

    return group

def load_factor_data(startDate = None):

    # Step 1: loading the dataframe and join them together
    # loading
    df_loadings = pd.read_parquet('john_rbics_industry.parquet')
    df_actual_returns = pd.read_parquet('john_return_prev_1d_open.parquet').rename(columns={"value": "return_1d"})
    df_sizes = pd.read_parquet('john_market_cap_open_dil.parquet').rename(columns={"value": "size"})
    # log and then winsorize
    df_sizes['size'] = np.log(df_sizes['size'])
    df_values = pd.read_parquet('john_market_cap_open_dil_to_net_income_ltm.parquet').rename(columns={"value": "value"})
    df_loadings = pd.get_dummies(df_loadings, columns=["industry"])

    # merge these dataframe into one
    df_values = df_values.merge(df_sizes, how='left', on=['id','datetime'])
    df_actual_returns = df_actual_returns.merge(df_values, how='left', on=['id','datetime'])
    df_loadings = pd.merge(df_actual_returns, df_loadings, how='left', on=None, left_on='id', right_on='id', left_index=False)

    for col in df_loadings.columns:
        if col.startswith("industry_"):
            df_loadings[col] = df_loadings[col].astype(int)

    # Step 2: Calculate momentum (e.g., 12-month cumulative return)

    # sliding window size for volatiltiy and momentum
    # look_back_period_volatility = 252  # Approx. 12 months for daily data
    look_back_period_momentum = 30
    # df_loadings['momentum'] = df_loadings.groupby('id')['return_1d'].apply(lambda x: x.rolling(window=look_back_period_momentum).apply(lambda y: np.prod(1 + y) - 1))
    # df_loadings['volatility'] = df_loadings.groupby('id')['return_1d'].apply(lambda x: x.rolling(window=look_back_period_volatility).std()).reset_index(level = 'id').rename(columns={"return_1d": "volatility"}).volatility
    df_loadings['momentum'] = df_loadings.groupby('id')['return_1d'].rolling(window=look_back_period_momentum).apply(
        lambda x: np.prod(1 + x) - 1, raw=True).reset_index(level='id').rename(
        columns={"return_1d": "momentum"}).momentum
    # drop those nan values less than then window size
    df_loadings['momentum'] = df_loadings['momentum'].shift(periods=1)
    df_loadings['country'] = 1

    df_loadings.dropna(inplace=True)
    df_loadings.replace([np.inf, -np.inf], 0, inplace=True)

    #Step 3: Winsorize and standardize
    if startDate:
        df_loadings = df_loadings[df_loadings.datetime >= startDate]


    cols = [col for col in df_loadings if
            col not in ['datetime', 'id', 'country', 'return_1d', 'period'] and not col.startswith('industry_')]

    # Apply the processing function to each group defined by 'datetime'
    df_loadings = df_loadings.groupby('datetime').apply(winsorization_standardization_group, cols = cols, include_groups=False)
    df_loadings = df_loadings.reset_index(level='datetime')

    return df_loadings
