import numpy as np
import pandas as pd 
import os
import pickle
import datetime
import time
from scipy import stats
import scipy.sparse as sp


# Frequency tables
def freq_standard(df, feature):
    """Number of bets in given league_id / number of bets place"""
    df = df.groupby(['customer_id', feature]).size().unstack(fill_value=0)
    # divide by sum
    df['sum'] = df.sum(axis=1)
    df = df.loc[:, df.columns != 'sum'].div(df["sum"], axis=0)
    return df

def freq_days(df, feature, n_dates):
    """Number of days that bet on given league_id over number of days in training set"""
    df = df.groupby(['customer_id', feature, 'date']).apply(lambda x: 1).unstack(fill_value=0)
    df['sum'] = df.sum(axis=1)
    df = df['sum'].unstack(fill_value=0)
    return df/n_dates

def get_sparsity(df):
    """Percentage sparsity of dataframe"""
    shape = df.shape 
    return sum(df.astype(bool).sum(axis=0))/(shape[0]*shape[1])

def log_transform(X, alpha, eps):
    def f(x):
        return 1+alpha*np.log(1+x/eps)
    Y = X.copy()
    Y.data = f(Y.data)
    return Y

def linear_transform(X, alpha):
    def f(x):
        return 1+alpha*x
    Y = X.copy()
    Y.data = f(Y.data)
    return Y

def sparse_to_list(X):
    """list of nonzero indices per row of sparse X"""
    result = np.split(X.indices, X.indptr)[1:-1]
    result = [list(r) for r in result]
    return result