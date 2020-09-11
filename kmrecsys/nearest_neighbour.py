import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import time

class NearestNeighbourSmall:

    def __init__(self, user_item_mat, user_index_map, item_index_map, on_user=True, ml=False):
        """
        Arguments:
        ----------
        user_item_mat : : `numpy.ndarray`
            (nUsers, nItems) matrix of `ratings'. Each row corresponds to a user point.
        n_neighbours : `int`
            Number of nearest neighbours to build model on
        on_user : `bool'
            If True uses user-user similarity, False calculates item-item
        ml : `bool'
            Indicates whether MovieLens dataset being used 
        """
        self.data = user_item_mat
        self.n_users, self.n_items = self.data.shape
        self.on_user = on_user
        self.ml = ml
        
        self.user_index_map = user_index_map
        self.item_index_map = item_index_map
        
        if on_user:
            self.S = cosine_similarity(self.data)
            self.m = self.n_users
        else:
            self.S = cosine_similarity(self.data.T)
            self.m = self.n_items

        self.top_sim = np.argsort(-self.S, axis=1)
        
        assert self.S.shape == (self.m, self.m)
        

    def fit(self, K, verbose=False):
        """
        Forms predicted ratings from K nearest neighbours.
        
        Parameters:
        -----------
        K : `int'
            Number of neighbours to compare, -1 indicates use all users/items as neighbours
        """
        self.R_hat = np.zeros(shape=(self.n_users, self.n_items))

        if K == -1:
            top_k_sim = self.top_sim[:,1:]
            # remove own user
            if self.on_user:
                self.R_hat = self.S @ self.data
            else:
                self.R_hat = self.data @ self.S
            self.R_hat = np.divide(self.R_hat - self.data, np.sum(self.S, axis=0)-np.ones(self.m))
        else:
            # ignore similarlity with same user/movie
            top_k_sim = self.top_sim[:,1:(K+1)]

            for i in range(self.m):
                s = self.S[i, top_k_sim[i]]
                if self.on_user:
                    R = self.data[top_k_sim[i], :]
                    self.R_hat[i,:] = (s @ R)/np.sum(s)
                else:
                    R = self.data[:, top_k_sim[i]]
                    self.R_hat[:,i] = (R @ s)/np.sum(s)
                    
                if verbose and i%500==0:
                    print('Item number {}'.format(i))
    
    def predict(self, userIDs, N):
        """
        Forms top N predictions for given userID ratings.

        Parameters:
        -----------
        userIDs : 
        
        N : `int'
            number of predictions to return
        """
        # all ratings sorted
        self.ranked_recs = np.argsort(-self.R_hat, axis=1)
        # top N ratings
        sorted_recs = np.array(self.ranked_recs[:,:N])
        
        f = lambda x: sorted_recs[self.user_index_map.index(x)].tolist()
        g = lambda x: [self.item_index_map[i] for i in x]
        
        recs = [g(f(i)) for i in userIDs]
        
        return recs


class NearestNeighbourBig:
    """
    Count frequencies per customer of each sport/major/league (choose one). 
    Predicts on most likely.
    """
    def __init__(self, user_item_mat, include_self=False, ml=False):
        """
        Arguments:
        ----------
        user_item_mat : : `numpy.ndarray`
            (nUsers, nItems) matrix of `ratings'. Each row corresponds to a user point.
        n_neighbours : `int`
            Number of nearest neighbours to build model on
        on_user : `bool'
            If True uses user-user similarity, False calculates item-item
        ml : `bool'
            Indicates whether MovieLens dataset being used 
        """
        
        self.data = user_item_mat
        
        self.n_users, self.n_items = self.data.shape
        self.ml = ml
        
        # self.user_index_map = user_index_map
        # self.item_index_map = item_index_map
        
        if include_self:
            self.i_0 = 0
        else:
            self.i_0 = 1
        

    def fit(self, K, verbose=False, verbose_at=10000):
        """
        Forms predicted ratings from K nearest neighbours.
        
        Parameters:
        -----------
        K : `int'
            Number of neighbours to compare, -1 indicates use all users/items as neighbours
        """
        self.K = K
        self.nbrs = NearestNeighbors(n_neighbors=K+1, algorithm='auto', metric='cosine').fit(self.data)

        self.R_hat = np.zeros(shape=(self.n_users, self.n_items))

        t0 = time.perf_counter()
        for i in range(self.n_users):
            distances, indices = self.nbrs.kneighbors(self.data[i], n_neighbors=self.K+1)

            s = distances.squeeze()[self.i_0:] # k x 1
            R = self.data[indices.squeeze()[self.i_0:]] # k x n_items
            self.R_hat[i,:] = (R.T @ s)/np.sum(s) # n_items x 1


            if verbose and i%verbose_at==0:
                t1 = time.perf_counter()
                print('Completed {} items in {:0.4f}'.format(i, t1-t0))
        
        t2 = time.perf_counter()
        print(f'Finished fitting in {t2 - t0:0.4f} R_hat ready')
    
    def recommend(self, user_ind, N):
        """
        Forms top N predictions for given userID ratings.

        Parameters:
        -----------
        userIDs : 
        
        N : `int'
            number of predictions to return
        """
        self.ranked_recs = np.argsort(-self.R_hat, axis=1)
        # self.mapped_IDs = [self.user_index_map.index(i) for i in userIDs]

        top_N_recs = np.array(self.ranked_recs[user_ind,:N])
        
        # g = lambda x: [self.item_index_map[i] for i in x]
        
        # recs = [g(top_N_recs[i].tolist()) for i in range(self.n_test)]
        # if self.ml:
        #     clf.R_hat[clf.R_hat < 3] = 0

        return top_N_recs.tolist()

    def recommend_all(self, N):
        """
        Forms top N predictions for given userID ratings.

        Parameters:
        -----------
        userIDs : 
        
        N : `int'
            number of predictions to return
        """
        self.ranked_recs = np.argsort(-self.R_hat, axis=1)
        top_N_recs = np.array(self.ranked_recs[:,:N])

        return top_N_recs.tolist()
 