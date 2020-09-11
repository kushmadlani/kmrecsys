import numpy as np
import pandas as pd 
import os
from utils import freq_days, freq_standard
from evaluation import pr_score

class Popularity:
    """
    Ranks events once by global frequency of bets
    """
    def __init__(self, user_item_mat):
        self.data = user_item_mat
        self.n_users, self.n_items = self.data.shape

    def fit(self):
        self.R_hat = np.repeat(np.array(self.data.sum(axis=0)), self.n_users, axis=0)

    def recommend(self, user_ind, N):
        r = np.argsort(-np.array(self.data.sum(axis=0)))
        self.ranked_recs = np.repeat(r, self.n_users, axis=0)
        return np.array(self.ranked_recs[user_ind,:N]).tolist()

    def recommend_all(self, N):
        r = np.argsort(-np.array(self.data.sum(axis=0)))
        self.ranked_recs = np.repeat(r, self.n_users, axis=0)
        return np.array(self.ranked_recs[:,:N]).tolist()
