import torch
from torch import nn
from recbole.model.abstract_recommender import SequentialRecommender

import numpy as np
from IPython import embed

from scipy import sparse
from scipy.sparse import csr_matrix, csc_matrix, vstack
from sklearn.preprocessing import normalize
from tqdm import tqdm
from time import time
import os

class SLIST(SequentialRecommender):
    def __init__(self, config, dataset):
        super(SLIST, self).__init__(config, dataset)
        self.device = config['device']
        self.logger = dataset.logger
        
        self.normalize = config['normalize'] if 'normalize' in config else 'l2'
        self.reg = config['reg'] if 'reg' in config else 10
        self.epsilon = config['epsilon'] if 'epsilon' in config else 100
        self.save_path = config['save_path'] if 'save_path' in config else None
        if self.save_path == 'None': self.save_path = None

        # for SLIT
        self.direction = config['direction'] if 'direction' in config else 'sr'
        self.train_weight = config['train_weight'] if 'train_weight' in config else 1.0
        # for SLIST
        self.alpha = config['alpha'] if 'alpha' in config else 0.5
        # for Dropout
        self.use_dropout = config['use_dropout'] if 'use_dropout' in config else False
        self.p = config['p'] if 'p' in config else 0.5

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
        # ||X - XB||
        input1, target1, row_weight1 = self.make_train_matrix(dataset, method='SLIS')

        # ||Y - ZB||
        input2, target2, row_weight2 = self.make_train_matrix(dataset, method='SLIT')
        # SLIST: alpha * ||X - XB|| + (1-alpha) * ||Y - ZB||
        input1.data = np.sqrt(self.alpha) * input1.data
        target1.data = np.sqrt(self.alpha) * target1.data
        input2.data = np.sqrt(1-self.alpha) * input2.data
        target2.data = np.sqrt(1-self.alpha) * target2.data

        input_matrix = vstack([input1, input2])
        target_matrix = vstack([target1, target2])
        w2 = row_weight1 + row_weight2  # list

        # P = (X^T * X + λI)^−1 = (G + λI)^−1
        # (A+B)^-1 = A^-1 - A^-1 * B * (A+B)^-1
        # P =  G
        self.logger.info(f"Train weight matrix...")
        train_start = G_start = time()
        W2 = sparse.diags(w2, dtype=np.float32)
        G = input_matrix.transpose().dot(W2).dot(input_matrix).toarray()
        G_time = time() - G_start
        self.logger.info(f"[{G_time:.2f}s] G is made. Sparsity:{(1 - np.count_nonzero(G)/(self.n_items**2))*100}%")

        P_start = time()
        if self.use_dropout:
            LAMBDA = np.diag((self.p / (1 - self.p)) * np.diag(G)) # dropout 
            LAMBDA += np.identity(self.n_items, dtype=np.float32) * self.reg # L2 regularization 
            P = np.linalg.inv(G + LAMBDA)
            del G, LAMBDA
        else:
            P = np.linalg.inv(G + np.identity(self.n_items, dtype=np.float32) * self.reg)
            del G
        P_time = time() - P_start
        self.logger.info(f"[{P_time:.2f}s] P is made (dropout:{self.use_dropout}, reg:{self.reg}, p:{self.p})")

        W_start = time()
        if self.epsilon < 10 and self.alpha == 1:
            C = -P @ (input_matrix.transpose().dot(W2).dot(input_matrix-target_matrix).toarray())

            mu = np.zeros(self.n_items)
            mu += self.reg
            mu_nonzero_idx = np.where(1 - np.diag(P)*self.reg + np.diag(C) >= self.epsilon)
            mu[mu_nonzero_idx] = (np.diag(1 - self.epsilon + C) / np.diag(P))[mu_nonzero_idx]

            # B = I - Pλ + C
            self.enc_w = np.identity(self.n_items, dtype=np.float32) - P @ np.diag(mu) + C
        else:
            self.enc_w = P @ input_matrix.transpose().dot(W2).dot(target_matrix).toarray()
        W_time = time() - W_start
        self.logger.info(f"[{W_time:.2f}s] weight matrix is made.")
        self.logger.info(f"[{time()-train_start:.2f}s] training is done")

        if self.save_path is not None:
            np.save(self.save_path, self.enc_w)
            print("weight matrix is saved")

    def make_train_matrix(self, dataset, method='SLIS'):
        if method == 'SLIS':
            input_row, input_col, input_data, target_row, target_col, target_data, w2 = self.get_sparse_slis(dataset)
        elif method == 'SLIT':
            input_row, input_col, input_data, target_row, target_col, target_data, w2 = self.get_sparse_slit(dataset)

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
        if self.normalize == 'l1':
            input_matrix = normalize(input_matrix, 'l1')
        elif self.normalize == 'l2':
            input_matrix = normalize(input_matrix, 'l2')
        else:
            pass

        return input_matrix, target_matrix, w2

    def get_sparse_slis_old(self, dataset):
        # load training data as numpy array
        session_id = dataset.inter_feat['session_id'].numpy() # [num_session, ]
        item_seq = dataset.inter_feat['item_id_list'].numpy() # [num_session, max_session_len]
        item_target = dataset.inter_feat['item_id'].numpy() # [num_session, ]
        session_len = dataset.inter_feat['item_length'].numpy() # [num_session, ]

        # for sparse matrix
        input_row, input_col, input_data = [], [], []
        target_row, target_col, target_data = [], [], []

        remain_items = []

        before_first_item = 0
        before_last_item = 0
        before_length = 0
        rowid = -1
        w2 = [] # row weight
        # make input matrix
        for i, (idx, seq, target, length) in enumerate(tqdm(zip(session_id, item_seq, item_target, session_len), total=len(session_id), desc='SLIS make train matrix')):
            # check if this session is augmentation of before session
            # case 1: before_first_item == seq[0] and before_last_item == target 
            if before_first_item == seq[0] and before_last_item == target and before_length == length + 1:
                is_augmented = True
            # case 2: next_length == length and next_target == last_item 
            elif i != len(session_id)-1 and session_len[i+1] == length and item_target[i+1] == seq[length-1]:
                remain_items.append(target)
                is_augmented = True
            else:
                is_augmented = False

            before_first_item = seq[0]
            before_last_item = seq[length-1]
            before_length = length

            if is_augmented:
                continue

            rowid += 1
            sessionitems = seq[:length].tolist() + [target]
            if len(remain_items) > 0:
                sessionitems = sessionitems + remain_items
                length += len(remain_items)
                remain_items = []
            w2.append(1) # TODO: current no weight in rebole !!
            slen = length + 1 # +1 for target item

            # add row to input matrix
            input_row += [rowid] * slen
            input_col += sessionitems

            before_first_item = seq[0]

        target_row = input_row
        target_col = input_col
        input_data = np.ones_like(input_row)
        target_data = np.ones_like(target_row)

        return input_row, input_col, input_data, target_row, target_col, target_data, w2

    def get_sparse_slis(self, dataset):
        # for sparse matrix
        input_row, input_col, input_data = [], [], []
        target_row, target_col, target_data = [], [], []
        w2 = [] # row weight
        
        # load training data
        full_session_path = f'dataset/{dataset.dataset_name}/{dataset.dataset_name}.train.session'
        token2id = dataset.field2token_id['item_id']
        with open(full_session_path, 'r') as f:
            rowid = -1
            for i, line in enumerate(f):
                if i == 0: continue # skip header
                line = line.strip().split('\t')
                rowid += 1
                sessionitems = list(map(int, line[1].split(' ')))
                sessionitems = [token2id[str(token)] for token in sessionitems] # convert token to id
                slen = len(sessionitems)
                
                input_row += [rowid] * slen
                input_col += sessionitems
                w2.append(1) # No comparable baselines use temporal information in our experiments
                
        # set target matrix (X) as input matrix (X)
        target_row = input_row
        target_col = input_col
        input_data = np.ones_like(input_row)
        target_data = np.ones_like(target_row)
        
        return input_row, input_col, input_data, target_row, target_col, target_data, w2

    def get_sparse_slit(self, dataset):
        # for sparse matrix
        input_row, input_col, input_data = [], [], []
        target_row, target_col, target_data = [], [], []
        w2 = [] # row weight
        
        # load training data
        full_session_path = f'dataset/{dataset.dataset_name}/{dataset.dataset_name}.train.session'
        with open(full_session_path, 'r') as f:
            rowid = -1
            for i, line in enumerate(f):
                if i == 0: continue # skip header
                line = line.strip().split('\t')
                sessionitems = list(map(int, line[1].split(' ')))
                sessionitems = [dataset.field2token_id['item_id'][str(token)] for token in sessionitems] # convert token to id
                slen = len(sessionitems)
                
                w2 += [1] * (slen-1) # No comparable baselines use temporal information in our experiments
                for t in range(slen-1):
                    rowid += 1
                    # set input matrix (S)
                    if self.direction == 'sr':
                        # t-th input item -> t+1~last target item
                        input_row += [rowid]
                        input_col += [sessionitems[t]]
                        input_data.append(0)
                    elif self.direction == 'all':
                        # first~t-th input item -> t+1~last target item
                        input_row += [rowid] * (t+1)
                        input_col += sessionitems[:t+1] 
                        for s in range(t+1):
                            input_data.append(-abs(t-s)) 
                    else:
                        raise ValueError(f"Invalid direction: {self.direction}")
                    
                    # set target matrix (T)
                    target_row += [rowid] * (slen - (t+1))
                    target_col += sessionitems[t+1:]
                    for s in range(t+1, slen):
                        target_data.append(-abs((t+1)-s))
                
        # Set value of input and target matrix
        input_data = list(np.exp(np.array(input_data) / self.train_weight))
        target_data = list(np.exp(np.array(target_data) / self.train_weight))

        return input_row, input_col, input_data, target_row, target_col, target_data, w2

    def forward(self, item_seq):
        '''
        item_seq: (B, L)
        e.g.,) item_seq = [[1,2,3,4,5,0,0,0,0,0], [1,2,3,4,5,6,7,8,9,10]]
        '''
        result = np.zeros((item_seq.shape[0], self.n_items), dtype=np.float32)

        # item_seq to one-hot array
        item_seq = item_seq.cpu().numpy()
        item_len = np.count_nonzero(item_seq, axis=1)
        for i in range(item_seq.shape[0]):
            session_items = item_seq[i][:item_len[i]]
            
            W_test = np.ones_like(session_items, dtype=np.float32) # [L]
            for j in range(len(W_test)):
                W_test[j] = np.exp(-abs(j+1-len(W_test)) / self.predict_weight)
            W_test = W_test.reshape(-1, 1) # [L, 1]

            result_i = self.enc_w[session_items] * W_test # [L, n_items]
            result_i = np.sum(result_i, axis=0) # [n_items]
            result[i] = result_i

        result = torch.from_numpy(result)

        return result

    def calculate_loss(self, interaction):
        loss = None
        return loss

    def predict(self, interaction):
        pass

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        result = self.forward(item_seq)

        return result

