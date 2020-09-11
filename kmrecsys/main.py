import numpy as np
import pandas as pd
import pickle
import scipy.sparse as sp
import implicit
import torch
from collections import defaultdict
import argparse
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from utils import log_transform, linear_transform, sparse_to_list
from evaluation import mapk, mr, m_p_r
from experiment import evaluate_model

import wandb


parser = argparse.ArgumentParser(description='Logistic Matrix Factorisation')
parser.add_argument('-d','--dataset', choices=['bets','lastfm'], required=True,
                    help='dataset to run experiment on (either lastfm or bets)')
parser.add_argument('-m','--model', choices=['imf','lmf','bpr'], required=True,
                    help='model to run experiment on, choice of Implicit Matrix Factorisation, Logistic Matrix Factorisation, Bayesian Personalised Ranking')
parser.add_argument('--top-n', type=int, default=5, metavar='top_N',
                    help='number of recommendations to make per user - to be evaluated used MAP@N and Recall')
parser.add_argument('-f','--factors', type=int, default=128, metavar='F',
                    help='latent dimension size')    
parser.add_argument('-lr', type=float, default=0.01,
                    help='learning rate to apply for updates')                        
parser.add_argument('--l2', type=float, default=0.01, metavar='l2',
                    help='L2 regularisation parameter')
parser.add_argument('--neg-prop', type=int, default=5,
                    help='proportion of negative samples')
parser.add_argument('--epochs', type=int, default=25, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--iterations_per_epoch', type=int, default=5, metavar='iter',
                    help='iterations of ALS per call of train')
parser.add_argument('-t', '--transform', choices=['log', 'linear', None], default=None,
                    help='choose data preprocessing')
parser.add_argument('--alpha', type=float, default=0.1,
                    help='linear scaling parameter in transformation')
parser.add_argument('--eps', type=float, default=10e-04,
                    help='log scaling parameter in transformation')
parser.add_argument('--project', type=str, required=True,
                    help='wandb project name')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('-t', '--transform', choices=['log', 'linear', None], default=None,
                    help='choose data preprocessing')
parser.add_argument('--alpha', type=float, default=0.1,
                    help='linear scaling parameter in transformation')
parser.add_argument('--eps', type=float, default=10e-04,
                    help='log scaling parameter in transformation')
parser.add_argument('--project', type=str, required=True,
                    help='wandb project name')
parser.add_argument('--test', type=str, default=0)

args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()
test_mode = True if args.test==1 else False

wandb.init(project=args.project)
wandb.config.update(args)

# load data 
if test_mode:
    raw_data = {
        'train': sp.load_npz('data/processed_'+args.dataset+'/full_train_mat.npz'),
        'test': sp.load_npz('data/processed_'+args.dataset+'/test_unmasked.npz'),
        'test_masked': sp.load_npz('data/processed_'+args.dataset+'/test_masked.npz') 
    }
else:
    raw_data = {
        'train': sp.load_npz('data/processed_'+args.dataset+'/train_ex_val.npz'),
        'val': sp.load_npz('data/processed_'+args.dataset+'/val_unmasked.npz'),
        'val_masked': sp.load_npz('data/processed_'+args.dataset+'/val_masked.npz') 
    }    

# transform data
if args.transform is 'log':
    data = {k: log_transform(v, args.alpha, args.eps) for k, v in raw_data.items()}
elif args.transform is 'linear':
    data = {k: linear_transform(v, args.alpha) for k, v in raw_data.items()}
else:
    data = raw_data

# instantiate item_user matrix
item_user = data['train'].T.tocsr()

# initialise model
if args.m == 'imf':
    model = implicit.als.AlternatingLeastSquares(factors=args.factors,
                                             regularization=args.l2,
                                             iterations=args.iterations_per_epoch,
                                             use_gpu=use_cuda,
                                             calculate_training_loss=False)
elif args.m == 'lmf':
    model = implicit.lmf.LogisticMatrixFactorization(factors=args.factors,
                                                 learning_rate=args.lr,
                                                 regularization=args.l2,
                                                 iterations=args.iterations_per_epoch,
                                                 neg_prop=args.neg_prop)
else: 
    model = implicit.bpr.BayesianPersonalizedRanking(factors=args.factors,
                                                 learning_rate=args.lr,
                                                 verify_negative_samples=True,
                                                 regularization=args.l2,
                                                 iterations=args.iterations_per_epoch,
                                                 use_gpu=use_cuda)

best_mpr = 1
for epoch in range(args.epochs):
    model.fit(item_user, show_progress=False)
    if args.m is not 'bpr':
        loss = implicit._als.calculate_loss(item_user.T.tocsr(), model.user_factors, model.item_factors, model.regularization)
        print('Epoch {}, Loss {}'.format(epoch, loss))

    if not test_mode:
        R_hat = model.user_factors@(model.item_factors.T)
        top_N_recs = np.array(np.argsort(-R_hat, axis=1)[:,:args.top_n]).tolist()
        MAP, rec_at_k, mpr_all, mpr_mask = evaluate_model(R_hat, top_N_recs, data['val'], data['val_masked'])
        del R_hat
        wandb.log({
            'MAP@N':MAP,
            'Recall@N':rec_at_k,
            'MPR (all)':mpr_all,
            'MPR (new)':mpr_mask
        })
        if mpr_mask<best_mpr:
            wandb.run.summary["best_mpr"] = mpr_mask
            best_mpr = mpr_mask

if test_mode:
    MAP, rec_at_k, mpr_all, mpr_mask = evaluate_model(R_hat, top_N_recs, data['test'], data['test_masked'])
    print('Final MPR of {}'.format(mpr_mask))
