import numpy as np
import pandas as pd


class AssetDigonalVarAdjuster:

    def __init__(self, df_loadings, df_factor_returns, window=None):
        """Initialization

        Parameters
        ----------
        df_loadings: pd.DataFrame
        df_factor_returns: pd.DataFrame

        Return
        ----------
        IVM (Idiosyncratic Variance Matrix (IVM): pd.DataFrame
         A matrix that contains the variances of the idiosyncratic (or specific) risks of different assets. This matrix is typically diagonal, where each diagonal element represents the variance of the idiosyncratic risk of an asset.
        """


        self._IVM = self.generate_idio_diagonal_matrix_df(df_loadings, df_factor_returns, window)

    @property
    def IVM(self):
        return self._IVM

    @property
    def datetimes(self):
        return self._first_level_index

    @property
    def ids(self):
        return self._second_level_index

    def generate_diagonal_matrix(self, datetime: str):
        # TODO
        return pd.DataFrame(np.diag(self._IVM.loc[datetime].variance),
                            index=self.ids,
                            columns=self.ids)

    def generate_idio_diagonal_matrix_df(self, df_loadings, df_factor_returns, window: int = None):
        # obtain factor columns as long as there is a return corresponding to the factor
        factor_cols = [
            col
            for col in df_loadings.columns if f"returns_{col}" in df_factor_returns.columns
            # if col in df_loadings.columns and col != "datetime"
        ]

        # Merge the DataFrames on 'date'
        df_idio = pd.merge(
            df_loadings, df_factor_returns, on="datetime", suffixes=("", "_factor_returns")
        ).drop(columns=[f"t_values_{col}" for col in factor_cols])  # drop factor t-stats

        # Multiplying matching columns
        for col in factor_cols:
            df_idio[f"calc_f_r_{col}"] = df_idio[col] * df_idio[f"returns_{col}"]

        # # Dropping the extra columns
        df_idio = df_idio.drop(columns=factor_cols)  # drop factor exposure columns
        df_idio = df_idio.drop(columns=[f"returns_{col}" for col in factor_cols])  # drop factor return columns

        df_idio["factor_return_1d"] = df_idio[[f"calc_f_r_{col}" for col in factor_cols]].sum(axis=1)
        df_idio["factor_return_1d_error"] = df_idio["factor_return_1d"] - df_idio["return_1d"]
        df_idio = df_idio[["datetime", "id", "return_1d", "factor_return_1d", "factor_return_1d_error"]]

        if window:
            # Calculate idiosync_return_var from return error
            df_idio.loc[:, "idio_risk_252d"] = (
                df_idio.sort_values(by=["id", "datetime"])
                .groupby("id")["factor_return_1d_error"]
                .rolling(window=window)
                .var()
                .reset_index(level=0, drop=True)
            )
        else:
            df_idio.loc[:, "idio_risk_252d"] = (
                df_idio.sort_values(by=["id", "datetime"])
                .groupby("id")["factor_return_1d_error"]
                .var()
                .reset_index(level=0, drop=True)
            )
        # print(df_idio.idio_risk_252d)
        # df_idio.dropna()

        df_idio = df_idio.set_index("datetime").pivot(columns="id", values="idio_risk_252d").fillna(0)
        self._first_level_index = df_idio.index
        self._second_level_index = df_idio.columns
        df_idio = df_idio.stack().reset_index()
        df_idio.columns = ['datetime', 'id', 'variance']
        df_idio.set_index(['datetime', 'id'], inplace=True)
        return df_idio