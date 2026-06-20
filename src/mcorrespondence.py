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

    