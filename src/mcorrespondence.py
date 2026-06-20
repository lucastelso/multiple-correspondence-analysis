import numpy as np
import pandas as pd
from scipy.linalg import svd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from typing import Literal

class ModernMCA(BaseEstimator, TransformerMixin):
    """
    Scikit-learn optimized Multiple Correspondence Analysis (MCA).
    Supports Benzécri and Greenacre corrections.
    """

    CORRECTIONS = Literal['none', 'benzecri', 'greenacre']

    def __init__(self, correction:CORRECTIONS='greenacre', n_components=2):
        """
        """
        self.n_components = n_components
        self.correction = str(correction).lower()

        # mathematical atributes
        self.eigenvalues_raw_ = None # Used for coordinates
        self.eigenvalues_corr_ = None # Used for explained variance
        self.total_inertia_ = None
        self.explained_variance_ration_ = None

        # Decompositio
        self.V_ = None
        self.U_ = None

        # Metadata
        self.column_masses_ = None
        self.row_masses_ = None
        self.categories_ = None

    def fit(self, X, y=None, sparse_output: bool = False):
        """
        Columns will be the feature names and the indexes will be the row names.
        Consider using sparse_output if you have a very big and dense matrix.
        """
        is_dataframe = isinstance(X, pd.DataFrame)
        if is_dataframe:
            self.feature_names_in_ = X.columns.to_list()
            self.row_names_ = X.index.to_list()
            X = X.values
        else:
            self.feature_names_in_ = None
            self.row_names_ = None
        
        self.encoder_ = OneHotEncoder(sparse_output=sparse_output, dtype=np.float64)
        Z = self.encoder_.fit_transform(X)

        if is_dataframe:
            self.categories_ = self.encoder_.get_feature_names_out(self.feature_names_in_)
        else:
            self.categories_ = self.encoder_.get_feature_names_out()

        total_N = np.sum(Z)
        P = Z / total_N

        r = np.sum(P, axis=1)
        c = np.sum(P, axis=0)
        self.column_masses_ = c
        self.row_masses_ = r

        expected = np.outer(r, c)
        denom = np.sqrt(np.outer(r, c))
        denom[denom == 0] = 1e-10 

        S = (P - expected) / denom

        # Dense SVD in NxJ matrix
        U, s, Vt = svd(S, full_matrices=False)

        # Raw Eigenvalues from burt's matrix (s^2 from indicator matrix) 
        lambda_raw = s ** 2
        self.eigenvalues_raw_ = lambda_raw.copy()

        self.n_components = min(self.n_components, len(s))

        Q = X.shape[1]
        J = Z.shape[1]
        average_inertia = 1.0 / Q

        if self.correction in ['benzecri', 'greenacre']:
            valid_idx = lambda_raw >= average_inertia
            lambda_valid = lambda_raw[valid_idx]

            # Corrected Eigenvalues (numerator is equal for Benzécri and Greenacre)
            lambda_corr = ((Q / (Q - 1.0)) * (lambda_valid - average_inertia)) ** 2

            # Fill the corrected eigenvalues in the final array (zero for invalid values)
            self.eigenvalues_corr_ = np.zeros_like(lambda_raw)
            self.eigenvalues_corr_[:len(lambda_corr)] = lambda_corr

            if self.correction == 'benzecri':
                # Benzécri: Total Inertia = sum of correction (optimistic)
                self.total_inertia_ = np.sum(lambda_corr)

            elif self.correction == 'greenacre':
                # Greenacre: Total inertia is the whole except for the off-diagonal values (rigorous)
                sum_sq_lambda = np.sum(lambda_raw ** 2)
                self.total_inertia_ = (Q / (Q - 1.0)) * (sum_sq_lambda - (J - Q) / (Q ** 2))
        
        else: # 'none' 
            self.eigenvalues_corr_ = lambda_raw.copy()
            self.total_inertia_ = np.sum(lambda_raw)

        # Truncates for the number chosen components
        self.eigenvalues_corr_ = self.eigenvalues_corr_[:self.n_components]
        self.explained_variance_ration_ = self.eigenvalues_corr_ / self.total_inertia_

        self.V_ = Vt.T[:, :self.n_components]
        self.U_ = U[:, :self.n_components]

        return self
    
    def get_row_stats(self):
        if self.U_ is None:
            raise ValueError("No trained model.")
        
        # Coordinates with raw eigenvalues
        scale_factor = np.sqrt(self.eigenvalues_raw_[:self.n_components])
        coords = self.U_ * scale_factor / np.sqrt(self.row_masses_)[:, None]

        cols = [f"Dim {i+1} ({self.explained_variance_ration_[i] * 100:.1f}%)" for i in self.n_components]
        index = self.row_names_ if self.row_names_ is not None else range(coords.shape[0])
        return pd.DataFrame(coords, index=index, columns=cols)
    
    def get_column_stats(self):
        if self.V_ is None:
            raise ValueError("No trained model.")
            
        # Coordinates with raw eigenvalues
        scale_factor = np.sqrt(self.eigenvalues_raw_[:self.n_components])
        coords = self.V_ * scale_factor / np.sqrt(self.column_masses_)[:, None]
        
        cols = [f"Dim {i+1}" for i in range(self.n_components)]
        df_coords = pd.DataFrame(coords, index=self.categories_, columns=cols)
        
        # Contributions with original metrics
        contrib = (self.column_masses_[:, None] * coords**2) / self.eigenvalues_raw_[:self.n_components]
        df_contrib = pd.DataFrame(contrib * 100, index=self.categories_, columns=[f"Contr {i+1}" for i in range(self.n_components)])
        
        return df_coords, df_contrib
    
    def transform(self, X):
        Z_new = self.encoder_.transform(X)
        Z_new = Z_new / np.sum(Z_new, axis=1)[:, None]
        
        col_coords, _ = self.get_column_stats()
        coords = Z_new @ col_coords.values
        
        # Transição usa autovalores CRUS
        for i in range(self.n_components):
            if self.eigenvalues_raw_[i] > 0:
                coords[:, i] /= np.sqrt(self.eigenvalues_raw_[i])
                
        return pd.DataFrame(coords, columns=[f"Dim {i+1}" for i in range(self.n_components)])