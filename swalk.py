import torch
from torch import nn

import numpy as np

from scipy import sparse
from scipy.sparse import csr_matrix, csc_matrix, vstack
from sklearn.preprocessing import normalize
from tqdm import tqdm
from time import time
import os

from slist import SLIST

class SWalk(SLIST):
    def __init__(self, config, dataset):
        # init parent of parent class
        super(SLIST, self).__init__(config, dataset)
        self.device = config['device']
        self.logger = dataset.logger
        
        self.normalize = config['normalize'] if 'normalize' in config else 'l2'
        self.reg = config['reg'] if 'reg' in config else 10
        self.epsilon = config['epsilon'] if 'epsilon' in config else 100
        self.save_path = config['save_path'] if 'save_path' in config else None
        if self.save_path == 'None': self.save_path = None

        self.save_load_SLIST = config['save_load_SLIST'] if 'save_load_SLIST' in config else False

        # for SLIT
        self.direction = config['direction'] if 'direction' in config else 'sr'
        self.train_weight = config['train_weight'] if 'train_weight' in config else 1.0
        # for SLIST
        self.alpha = config['alpha'] if 'alpha' in config else 0.5
        # for Dropout
        self.use_dropout = config['use_dropout'] if 'use_dropout' in config else False
        self.dropout_p = config['dropout_p'] if 'dropout_p' in config else 0.5

        # for SWalk
        self.model_transition = config['model_transition'] if 'model_transition' in config else 'SLIT'
        self.model_teleportation = config['model_teleportation'] if 'model_teleportation' in config else 'SLIS'
        self.self_beta = config['self_beta'] if 'self_beta' in config else 1.0

        self.walk_p = config['walk_p'] if 'walk_p' in config else 0.5
        # self.PR_eps = config['PR_eps'] if 'PR_eps' in config else 100
        self.PR_eps = 0.01 * self.n_items

        # for inference
        self.predict_weight = config['predict_weight'] if 'predict_weight' in config else 2.0

        # for test
        self.enc_w = np.ones((self.n_items, self.n_items), dtype=np.float32)
        self.dummy_param = nn.Parameter(torch.Tensor(1, 1)) # for using self.parameters()

        if self.save_path is not None:
            if self.save_path[-4:] != '.npy':
                self.save_path = self.save_path + '.npy'

            if os.path.exists(self.save_path):
                self.enc_w = np.load(self.save_path)

                if self.enc_w.shape != (self.n_items, self.n_items):
                    print("weight matrix is not loaded")
                else:
                    print("weight matrix is loaded")
                    return
            print(f"wrong save_path or file is not exist.({self.save_path}) Train weight matrix...")
        else:
            print("No save_path")
        
        self.fit(config, dataset)
    
    def fit(self, config, dataset):
        train_start = time()
        self.logger.info(f"Train weight matrix...")

        if self.model_transition in ['SLIS', 'SLIT']:
            model_transition = self.make_SLIST(dataset, method=self.model_transition)
        elif self.model_transition == 'I':
            model_transition = np.diag(np.ones(self.n_items))

        if self.model_teleportation in ['SLIS', 'SLIT']:
            model_teleportation = self.make_SLIST(dataset, method=self.model_teleportation)
        elif self.model_teleportation == 'I':
            model_teleportation = np.diag(np.ones(self.n_items))

        self.enc_w = self.do_random_walk(S=model_transition, T=model_teleportation)

        self.logger.info(f"[{time()-train_start:.2f}s] training is done")

        if self.save_path is not None:
            np.save(self.save_path, self.enc_w)
            print("weight matrix is saved")
    
    def make_SLIST(self, dataset, method):
        SLIST_path = f'./saved/{dataset.dataset_name}_{method}.npy'
        if self.save_load_SLIST and os.path.exists(SLIST_path):
            print(f"Load {method} matrix")
            return np.load(SLIST_path)

        input_matrix, target_matrix, w2 = self.make_train_matrix(dataset, method=method)

        # P = (X^T * X + λI)^−1 = (G + λI)^−1
        # (A+B)^-1 = A^-1 - A^-1 * B * (A+B)^-1
        # P =  G
        train_start = G_start = time()
        W2 = sparse.diags(w2, dtype=np.float32)
        G = input_matrix.transpose().dot(W2).dot(input_matrix).toarray()
        G_time = time() - G_start
        print(f"[{G_time:.2f}s] G is made. Sparsity:{(1 - np.count_nonzero(G)/(self.n_items**2))*100}%")

        P_start = time()
        if self.use_dropout:
            LAMBDA = np.diag((self.dropout_p / (1 - self.dropout_p)) * np.diag(G)) # dropout 
            LAMBDA += np.identity(self.n_items, dtype=np.float32) * self.reg # L2 regularization 
            P = np.linalg.inv(G + LAMBDA)
            del G, LAMBDA
        else:
            P = np.linalg.inv(G + np.identity(self.n_items, dtype=np.float32) * self.reg)
            del G
        P_time = time() - P_start
        print(f"[{P_time:.2f}s] P is made (dropout:{self.use_dropout}, reg:{self.reg}, dropout_p:{self.dropout_p})")

        W_start = time()
        enc_w = P @ input_matrix.transpose().dot(W2).dot(target_matrix).toarray()
        W_time = time() - W_start
        print(f"[{W_time:.2f}s] weight matrix is made.")
        print(f"[{time()-train_start:.2f}s] training is done")

        if self.save_load_SLIST:
            np.save(SLIST_path, enc_w)
            print(f"Save {method} matrix")

        return enc_w

    def do_random_walk(self, S, T):
        # non-negative matrix
        S[S < 0] = 0
        T[T < 0] = 0

        # For efficiency, we use sparse matrix
        S = csr_matrix(S)
        T = csr_matrix(T)

        # S = Diag(W)^-1 \cdot W
        S = normalize(S, norm='l1', axis=1)
        T = normalize(T, norm='l1', axis=1)

        S = csr_matrix(S)
        T = csr_matrix(T)

        T = self.self_beta*T + csr_matrix((1-self.self_beta)*np.diag(np.ones(self.n_items)))

        print(f'Sparsity of S: {1 - (S.count_nonzero()/(S.shape[0]*S.shape[1]))}')
        print(f'Sparsity of T: {1 - (T.count_nonzero()/(T.shape[0]*T.shape[1]))}')

        # recwalk_method => 'PR'
        # Sigma(0~inf) (1-p)*p^k*S^k == 1-p*1*I + Sigma(1~inf) (1-p)*p^k*S^k
        M = sparse.diags(np.ones(self.n_items), format='csr')

        # recwalk_dense
        M = M.toarray()
        S = S.toarray()
        T = T.toarray()

        k_step = 100
        for _ in tqdm(range(k_step), desc='doing random walk'):
            M_last = M.copy()
            M = self.walk_p * (M @ S) + (1-self.walk_p)*T

            # Check converge, l1 norm
            err = abs(M_last - M).sum()
            if err < self.PR_eps:
                print(f'err: {err} < {self.PR_eps}')
                break
            print(f'err: {err} > {self.PR_eps}')
        
        final_matrix = M   

        return final_matrix