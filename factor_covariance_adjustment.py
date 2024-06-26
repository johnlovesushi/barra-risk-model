import numpy as np
import pandas as pd
from bias_statistics import BiasStatsCalculator
from utils import cov_ewa


class FactorCovAdjuster:
    """Adjustments on factor covariance matrix"""

    def __init__(self, FRM: pd.DataFrame, window: int | None = None) -> None:
        """Initialization

        Parameters
        ----------
        FRM : np.ndarray
            Factor return matrix (T*K)
        """
        self.T, self.K = FRM.shape
        if self.K > self.T:
            raise Exception("number of periods must be larger than number of factors")
        self.FRM = FRM.astype("float64")

        if window and window > self.T:
            raise Exception("number of window must be larger than number of periods")
    
        if window:

            self.window = window
            self.first_level_index = FRM.index[self.window-1:]
        else:
            self.first_level_index = FRM.index
            self.window = 0
            
        self.FCM = None
        self.second_level_index = FRM.columns
        self.first_level_index = FRM.index[self.window-1:]

    def convert_cov_matrices(self,cov_matrices: list) -> pd.DataFrame:
        """Convert the list of covariance matrix (3D) to 2D pandas DataFrame

        Parameters
        ----------
        cov_matrices : list
            A list of covariance matrix (one or more depends on the window)

        Returns
        -------
        pd.DataFrame
            cov_matrices, 2D pandas.DataFrame with 2 level of index (datetime, factor)
        """
        cov_matrices = np.array(cov_matrices)
        multi_index = pd.MultiIndex.from_product([self.first_level_index, self.second_level_index], names=['datetime', 'factor'])
        return pd.DataFrame(cov_matrices.reshape((-1, cov_matrices.shape[-1])), index = multi_index, columns = self.second_level_index)
        
        
    def calc_fcm_raw(self, half_life: int | None = None) -> pd.DataFrame:
        """Calculate the factor covariance matrix, FCM (K*K)

        Parameters
        ----------
        half_life : int
            Steps it takes for weight in EWA to reduce to half of the original value

        Returns
        -------
        np.ndarray
            FCM, denoted by `F_Raw`
        """
        cov_matrices = []
        if not self.window:
            cov_matrices.append(cov_ewa(self.FRM.T.to_numpy(), half_life))
        else:
            for i in range(len(self.FRM)-self.window+1):
                cov_matrices.append(cov_ewa(self.FRM.iloc[i:i+self.window,:].T.to_numpy(), half_life))

        return self.convert_cov_matrices(cov_matrices)

    def newey_west_adjust(self,
        FRM: np.ndarray, half_life: int, max_lags: int, multiplier: int
    ) -> np.ndarray:
        """Apply Newey-West adjustment on `F_Raw`

        Parameters
        ----------
        half_life : int
            Steps it takes for weight in EWA to reduce to half of the original value
        max_lags : int
            Maximum Newey-West correlation lags
        multiplier : int
            Number of periods a FCM with new frequence contains

        Returns
        -------
        np.ndarray
            Newey-West adjusted FCM, denoted by `F_NW`
        """
        FCM = 0
        for D in range(1, max_lags + 1):
            C_pos_delta = cov_ewa(FRM, half_life, D)
            FCM += (1 - D / (1 + max_lags)) * (C_pos_delta + C_pos_delta.T)
        D, U = np.linalg.eigh(FCM * multiplier)
        D[D <= 0] = 1e-14  # fix numerical error
        self.FCM = U.dot(np.diag(D)).dot(U.T)
        D, U = np.linalg.eigh(FCM)
        return FCM

    def calc_newey_west_frm(self, max_lags:int, multiplier: int, half_life: int | None = None) -> pd.DataFrame:

        cov_matrices = []
        if not self.window:
            cov_matrices.append(self.newey_west_adjust(self.FRM.T.to_numpy(), 
                                                  half_life = half_life, 
                                                  max_lags = max_lags, 
                                                  multiplier = multiplier))
        else:
            for i in range(len(self.FRM)-self.window+1):
                cov_matrices.append(self.newey_west_adjust(self.FRM.iloc[i:i+self.window,:].T.to_numpy(),
                                            half_life = half_life,
                                            max_lags = max_lags, 
                                            multiplier = multiplier))

        return self.convert_cov_matrices(cov_matrices)

    def eigenfactor_risk_adjust(self, FCM, coef: float, M: int = 1000) -> np.ndarray:
        """Apply eigenfactor risk adjustment on `F_NW`

        Parameters
        ----------
        coef : float
            Adjustment coefficient
        M : int, optional
            Times of Monte Carlo simulation, by default 1000

        Returns
        -------
        np.ndarray
            Eigenfactor risk adjusted FCM, denoted by `F_Eigen`
        """
        D_0, U_0 = np.linalg.eigh(FCM)
        D_0[D_0 <= 0] = 1e-14  # fix numerical error
        Lambda = np.zeros((self.K,))
        for _ in range(M):
            b_m = np.array([np.random.normal(0, d**0.5, self.T) for d in D_0])
            f_m = U_0.dot(b_m)
            F_m = f_m.dot(f_m.T) / (self.T - 1)
            D_m, U_m = np.linalg.eigh(F_m)
            D_m[D_m <= 0] = 1e-14  # fix numerical error
            D_m_tilde = U_m.T.dot(FCM).dot(U_m)
            Lambda += np.diag(D_m_tilde) / D_m
        Lambda[Lambda <= 0] = 1e-14  # fix numerical error
        Lambda = np.sqrt(Lambda / M)
        Gamma = coef * (Lambda - 1.0) + 1.0
        D_0_tilde = Gamma**2 * D_0
        FCM = U_0.dot(np.diag(D_0_tilde)).dot(U_0.T)
        return FCM

    def calc_eigenfactor_risk_frm(self, max_lags:int, multiplier: int, half_life: int | None = None, 
                                  coef: float = 1.2, M: int = 1000) -> pd.DataFrame:

        cov_matrices = []
        if not self.window:
            FCM = self.newey_west_adjust(self.FRM.T.to_numpy(), 
                                                  half_life = half_life, 
                                                  max_lags = max_lags, 
                                                  multiplier = multiplier)
            cov_matrices.append(self.eigenfactor_risk_adjust(FCM, coef, M))
        else:
            for i in range(len(self.FRM)-self.window+1):
                FCM = self.newey_west_adjust(self.FRM.iloc[i:i+self.window,:].T.to_numpy(),
                                            half_life = half_life,
                                            max_lags = max_lags, 
                                            multiplier = multiplier)
                cov_matrices.append(self.eigenfactor_risk_adjust(FCM, coef, M))

        return self.convert_cov_matrices(cov_matrices)

    def volatility_regime_adjust(self, FCM: np.ndarray, FRM: np.ndarray, half_life: int) -> np.ndarray:
        """Apply volatility regime adjustment on `F_Eigen`

        Parameters
        ----------
        FCM : np.ndarray
            Previously estimated factor covariance matrix (last `F_Eigen`, since `F_VRA`
            could lead to huge fluctuations) on only one period (not aggregated); the
            order of factors should remain the same
        half_life : int
            Steps it takes for weight in EWA to reduce to half of the original value

        FRM: np.ndarray
            Factor return matrix (T*K)
        Returns
        -------
        np.ndarray
            Volatility regime adjusted FCM, denoted by `F_VRA`
        """
        sigma = np.sqrt(np.diag(FCM))
        B = BiasStatsCalculator(FRM, sigma).single_window(half_life)
        FCM = FCM * (B**2).mean(axis=0)  # Lambda^2
        return FCM

    def calc_volatility_regime_frm(self, max_lags:int, multiplier: int, half_life: int | None = None, 
                                  coef: float = 1.2, M: int = 1000):

        cov_matrices = []
        if not self.window:
            FCM = self.newey_west_adjust(self.FRM.T.to_numpy(), 
                                                  half_life = half_life, 
                                                  max_lags = max_lags, 
                                                  multiplier = multiplier)
            
            FCM = self.eigenfactor_risk_adjust(FCM, coef, M)
            cov_matrices.append(self.volatility_regime_adjust(FCM, self.FRM.T.to_numpy(),half_life))
        else:
            for i in range(len(self.FRM)-self.window+1):
                FCM = self.newey_west_adjust(self.FRM.iloc[i:i+self.window,:].T.to_numpy(),
                                            half_life = half_life,
                                            max_lags = max_lags, 
                                            multiplier = multiplier)
                
                FCM = self.eigenfactor_risk_adjust(FCM, coef, M)
                cov_matrices.append(self.volatility_regime_adjust(FCM, self.FRM.iloc[i:i+self.window,:].T.to_numpy(),half_life))

        return self.convert_cov_matrices(cov_matrices)    