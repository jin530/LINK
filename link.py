import torch
from torch import nn
import torch.nn.functional as F
from recbole.model.layers import TransformerEncoder
from recbole.model.abstract_recommender import SequentialRecommender

from core_ave import COREave
from IPython import embed
from entmax import entmax_bisect

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils_seq import reverse_packed_sequence
import numpy as np
import math

from scipy import sparse
from scipy.sparse.linalg import inv
from scipy.sparse import csr_matrix, csc_matrix, vstack
from sklearn.preprocessing import normalize
from tqdm import tqdm
from time import time
import os
import pickle

from slist import SLIST

class LINK(SLIST):
    def __init__(self, config, dataset):
        super(SLIST, self).__init__(config, dataset)
        self.device = config['device']
        self.logger = dataset.logger
        
        self.reg = config['reg'] if 'reg' in config else 10.0
        self.save_path = config['save_path'] if 'save_path' in config else None

        if self.save_path == 'None': self.save_path = None

        self.data_name = dataset.dataset_name

        # for Neural Teacher (or SLIT)
        self.reg_teacher = config['reg_teacher'] if 'reg_teacher' in config else 10.0
        self.teacher_path = config['teacher_path'] if 'teacher_path' in config else None
        self.direction = config['direction'] if 'direction' in config else 'sr'
        self.train_weight = config['train_weight'] if 'train_weight' in config else 1.0
        self.teacher_temperature = config['teacher_temperature'] if 'teacher_temperature' in config else 1.0

        # for extended SLIS
        self.save_load_SLIST = config['save_load_SLIST'] if 'save_load_SLIST' in config else False
        self.extend_beta = config['extend_beta'] if 'extend_beta' in config else 1.0
        self.slis_matrix = None

        # for closed_form aggregation
        self.teacher_normalize = config['teacher_normalize'] if 'teacher_normalize' in config else True
        self.slis_alpha = config['slis_alpha'] if 'slis_alpha' in config else 0.5

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
            print(f"No save_path for Full Model ({self.__class__.__name__})")
            
        self.fit(config, dataset)
    
    def fit(self, config, dataset):
        # Make SLIS
        print(f"Make SLIS matrix for LIS")
        self.slis_matrix = self.make_SLIST(dataset, method='SLIS')

        # ||X - XB|| + ||X^aug - X^augB||
        input1, target1, row_weight1 = self.make_train_matrix(dataset, method='LIS', normalize_method='l1')
        # ||T - SB||
        input2, target2, row_weight2 = self.make_train_matrix(dataset, method='SLIT', normalize_method='l1')
        # T^M
        teacher_matrix = np.load(self.teacher_path)

        # Obj: alpha * ||X - XB|| + alpha * ||X^aug - X^augB||  + (1-alpha) * ||T- SB|| + beta * ||T^M-B|| + λ||B||
        norm_start = time()
        if self.teacher_normalize:
            # softmax for teacher
            teacher_matrix = torch.from_numpy(teacher_matrix).cpu()
            teacher_matrix = F.softmax(teacher_matrix / self.teacher_temperature, dim=1).numpy()
            teacher_matrix = teacher_matrix.astype(np.float32)
        print(f"[{time()-norm_start:.2f}s] normalization is done")

        input1.data = np.sqrt(self.slis_alpha) * input1.data
        target1.data = np.sqrt(self.slis_alpha) * target1.data
        input2.data = np.sqrt(1 - self.slis_alpha) * input2.data
        target2.data = np.sqrt(1 - self.slis_alpha) * target2.data

        input_matrix = vstack([input1, input2])
        target_matrix = vstack([target1, target2])
        w2 = row_weight1 + row_weight2

        print(f"input_matrix: {input_matrix.shape}, target_matrix: {target_matrix.shape}")
        print(f"slis_alpha: {self.slis_alpha}, reg_teacher:{self.reg_teacher}")
        print(f"extend_beta: {self.extend_beta}")

        # Obj: ||T' - S'B|| + beta||T^M-B|| + λ||B||
        # (T' = concat(X, X^aug, T))
        # (S' = concat(X, X^aug, S))
        
        # B = (T^T @ T + (beta + λ)I)^−1 @ (T^T @ S + beta*T^M)
        #   = (G + (beta + λ)I)^−1 @ (T^T @ S + beta*T^M) ; G = X^T @ X
        #   = P @ (T^T @ S + beta*T^M) ; P = (G + (beta + λ)I)^−1
        self.logger.info(f"Train weight matrix...")
        train_start = G_start = time()
        W2 = sparse.diags(w2, dtype=np.float32)
        G = input_matrix.transpose().dot(W2).dot(input_matrix).toarray()
        G_time = time() - G_start
        self.logger.info(f"[{G_time:.2f}s] G is made. Sparsity:{(1 - np.count_nonzero(G)/(self.n_items**2))*100}%")

        P_start = time()
        LAMBDA = self.reg_teacher * np.identity(self.n_items, dtype=np.float32)
        P = np.linalg.inv(G + LAMBDA)
        P_time = time() - P_start
        self.logger.info(f"[{P_time:.2f}s] P is made")

        W_start = time()
        print(f"closed-form aggregation (reg_teacher: {self.reg_teacher})")
        self.enc_w = P @ (input_matrix.transpose().dot(W2).dot(target_matrix).toarray() + self.reg_teacher * teacher_matrix)
        W_time = time() - W_start
        self.logger.info(f"[{W_time:.2f}s] weight matrix is made.")
        self.logger.info(f"[{time()-train_start:.2f}s] training is done")

        if self.save_path is not None:
            np.save(self.save_path, self.enc_w)
            print("weight matrix is saved")
            
    def make_SLIST(self, dataset, method):
        SLIST_path = f'./saved/{dataset.dataset_name}_{method}.npy'
        if self.save_load_SLIST and os.path.exists(SLIST_path):
            print(f"Load {method} matrix")
            return np.load(SLIST_path)
        else:
            print(f"{method} matrix is not loaded. It will be made and saved to {SLIST_path}")

        input_matrix, target_matrix, w2 = self.make_train_matrix(dataset, method=method, normalize_method='l2')

        # P = (X^T * X + λI)^−1 = (G + λI)^−1
        # (A+B)^-1 = A^-1 - A^-1 * B * (A+B)^-1
        # P =  G
        train_start = G_start = time()
        W2 = sparse.diags(w2, dtype=np.float32)
        G = input_matrix.transpose().dot(W2).dot(input_matrix).toarray()
        G_time = time() - G_start
        print(f"[{G_time:.2f}s] G is made. Sparsity:{(1 - np.count_nonzero(G)/(self.n_items**2))*100}%")
    
        P_start = time()
        P = np.linalg.inv(G + np.identity(self.n_items, dtype=np.float32) * self.reg)
        del G
        P_time = time() - P_start
        print(f"[{P_time:.2f}s] P is made (reg:{self.reg})")

        W_start = time()
        enc_w = P @ input_matrix.transpose().dot(W2).dot(target_matrix).toarray()
        W_time = time() - W_start
        print(f"[{W_time:.2f}s] weight matrix is made.")
        print(f"[{time()-train_start:.2f}s] training is done")

        if self.save_load_SLIST:
            np.save(SLIST_path, enc_w)
            print(f"Save {method} matrix")

        return enc_w

    def make_train_matrix(self, dataset, method='SLIS', normalize_method='X'):
        if method == 'SLIS':
            input_row, input_col, input_data, target_row, target_col, target_data, w2 = self.get_sparse_slis(dataset)
        if method == 'SLIT':
            input_row, input_col, input_data, target_row, target_col, target_data, w2 = self.get_sparse_slit(dataset)
        if method == 'LIS':
            input_row, input_col, input_data, target_row, target_col, target_data, w2 = self.get_sparse_full_kd_slis(dataset)

        # Make sparse_matrix
        mat_start = time()
        input_matrix = csr_matrix((input_data, (input_row, input_col)), shape=(max(input_row)+1, self.n_items), dtype=np.float32)
        target_matrix = csr_matrix((target_data, (target_row, target_col)), shape=input_matrix.shape, dtype=np.float32)
        print(f"[{method}] input sparse matrix {input_matrix.shape} is made.  Sparsity:{(1 - input_matrix.count_nonzero()/(self.n_items*input_matrix.shape[0]))*100:.5f}%")
        print(f"[{method}] target sparse matrix {target_matrix.shape} is made.  Sparsity:{(1 - target_matrix.count_nonzero()/(self.n_items*target_matrix.shape[0]))*100:.5f}%")
        mat_time = time() - mat_start
        print(f"[{method}, {mat_time:.2f}s] matrix is made.")

        if method == 'SLIS':
            # Clip Value of repeated items to 1
            input_matrix.data = np.minimum(input_matrix.data, 1)
            target_matrix.data = np.minimum(target_matrix.data, 1)

        # Normalization
        if normalize_method == 'l1':
            input_matrix = normalize(input_matrix, 'l1')
        elif normalize_method == 'l2':
            input_matrix = normalize(input_matrix, 'l2')
        else:
            pass
        
        return input_matrix, target_matrix, w2

    def get_sparse_full_kd_slis(self, dataset):
        assert self.slis_matrix is not None, "slis_matrix is not loaded"
        
        # for sparse matrix
        input_row, input_col, input_data = [], [], []
        target_row, target_col, target_data = [], [], []
        w2 = []
        
        num_session_item_all = 0
        num_kd_item_all = 0
        
        # load training data
        full_session_path = f'dataset/{dataset.dataset_name}/{dataset.dataset_name}.train.session'
        with open(full_session_path, 'r') as f:
            rowid = -1
            for i, line in enumerate(tqdm(f, desc='Loading training data', dynamic_ncols=True)):
                if i == 0: continue # skip header
                line = line.strip().split('\t')
                rowid += 1
                sessionitems = list(map(int, line[1].split(' ')))
                sessionitems = [dataset.field2token_id['item_id'][str(token)] for token in sessionitems] # convert token to id
                
                # Clip Value of repeated items to 1
                sessionitems = list(set(sessionitems))
                slen = len(sessionitems)
                
                # Prediction using SLIS
                output = self.slis_matrix[sessionitems] # [slen, n_items]
                output = np.sum(output, axis=0) # [n_items]
                output[output < 0] = 0 # clip negative values
                
                num_kd_items = min(slen * 9, 1000) # max 1000 items
                
                # get top-k items (using torch)
                output_t = torch.from_numpy(output) # [n_items]
                topk_scores, topk_indices = torch.topk(output_t, num_kd_items, sorted=False) # [num_kd_items]
                topk_scores = topk_scores.numpy()
                topk_indices = topk_indices.tolist()
                
                # add row to input matrix (original)
                input_row += [rowid] * slen
                input_col += sessionitems
                input_data += [1 * (1-self.extend_beta)] * slen # (sum = slen * (1-beta))
                
                # add row to input matrix (predicted)
                input_row += [rowid] * num_kd_items
                input_col += topk_indices
                sum_kd_scores = np.sum(topk_scores)
                input_data += [kd_score / sum_kd_scores * self.extend_beta for kd_score in topk_scores] # (sum = slen * beta)
                
                w2.append(1) # No comparable baselines use temporal information in our experiments
                
                num_session_item_all += slen
                num_kd_item_all += num_kd_items
                
        print(f"avg. num_session_item: {num_session_item_all / (rowid+1):.2f}")
        print(f"avg. num_kd_item: {num_kd_item_all / (rowid+1):.2f}")
        
        target_row = input_row
        target_col = input_col
        target_data = input_data
        
        return input_row, input_col, input_data, target_row, target_col, target_data, w2