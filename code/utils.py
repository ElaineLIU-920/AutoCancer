
import torch
import math
import pickle
import random
import numpy as np
import torch.autograd as autograd
import torch.nn.functional as F
import pandas as pd
import torch.optim as optim

from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset

from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc
from imblearn.over_sampling import RandomOverSampler

from scipy.special import softmax

from math import ceil
from copy import deepcopy

import os
import gc
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

seed = 1105
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
###############################################
#--------------   Dataset tools   ------------#
###############################################
def flatten2array(multiindex):
    array_2d = []
    for i in range(multiindex.shape[0]):
        temp1 = []
        for j in multiindex.iloc[i]:
            if (type(j) == np.ndarray or type(j) == list):
                for k in j:
                    temp1.append(k)
            else:
                temp1.append(j)
        array_2d.append(temp1)
    array_2d = np.array(array_2d) 
    return array_2d

def TruncatePadding(snv_array, max_variats):
    
    """
    Function description: truncate or padding variable-length variants
    
    """
    if len(snv_array) >= max_variats:
        return snv_array[:max_variats]  # truncate
    else:
        return np.append(snv_array,np.zeros((max_variats - snv_array.shape[0],snv_array.shape[1])), axis=0) # padding with zero array

def len_feature(x):
    if x is None:
        length = 0
    elif isinstance(x.index, pd.MultiIndex):
        length = len(x.index.get_level_values(x.index.names[0]).unique())
    else:
        length = len(x)
    return length

class FeatureSelectionDataset(Dataset):
    def __init__(self, x_1d=None, x_2d=None, y=None, max_len_2d=84):
        super().__init__()
        self.x_1d, self.x_2d, self.y = x_1d, x_2d, y
        len_1d, len_2d = len_feature(self.x_1d), len_feature(self.x_2d)
        self.len, self.max_len_2d = max(len_1d, len_2d), max_len_2d

    def __getitem__(self, index):
        if (self.x_1d is not None)&(self.x_2d is not None):
            k = self.x_1d.index[index]
            if k in self.x_2d.index.get_level_values(self.x_2d.index.names[0]).unique():
                x_2d = flatten2array(self.x_2d.loc[k])
            else:
                x_2d = np.zeros(flatten2array(self.x_2d[:1]).shape)
            x_2d = TruncatePadding(x_2d, self.max_len_2d)
            return np.array(self.x_1d.loc[k].values.tolist()), x_2d, self.y[k]
        elif (self.x_2d is not None):
            k = self.y.index[index]
            if k in self.x_2d.index.get_level_values(self.x_2d.index.names[0]).unique():
                x_2d = flatten2array(self.x_2d.loc[k])
            else:
                x_2d = np.zeros(flatten2array(self.x_2d[:1]).shape)
            x_2d = TruncatePadding(x_2d, self.max_len_2d)
            return [], x_2d, self.y[k]
        elif self.x_1d is not None:
            k = self.x_1d.index[index]
            return np.array(self.x_1d.loc[k].values.tolist()), [], self.y[k]
        else:
            print('No data provided!')
            return None
    def __len__(self):
        return self.len

###############################################
#---------------   Model define   ------------#
###############################################
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask=None):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)  ## (b,h,l,d) * (b,h,d,l)
    if mask is not None:
        # scores = scores.masked_fill(mask == 0, -1e9)
        scores = scores.masked_fill(mask, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn   ##(b,h,l,l) * (b,h,l,d) = (b,h,l,d)

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super().__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + autograd.Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)



class AdaptiveAttentionModel(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, d_model, d_ff, head, num_layer, dropout):
        super().__init__()
        attn = MultiHeadedAttention(head, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        layer = EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), dropout)
        self.layers = clones(layer, num_layer)
        self.norm = LayerNorm(layer.size)
        self.posi = PositionalEncoding(d_model, dropout)
        self.out = nn.Linear(d_model, 1) 
        # self.attn_matrices = []

    def forward(self, x):
        "Pass the input (and mask) through each layer in turn."
        # x: embedding (b,l,we)
        x = self.posi(x)
        for layer in self.layers:
            x = layer(x)
            # self.attn_matrices.append(layer.self_attn.attn)
        return self.out(self.norm(x))

class AdaptiveMLP(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, num_fc_input, num_output_nodes, num_fc_layers, initial_dropout, act=nn.Sigmoid()):
        super().__init__()
        self.num_fc_layers, self.act = num_fc_layers, act
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if num_fc_layers == 1:
            self.fc0 = nn.Linear(num_fc_input, num_output_nodes)
        else:
            # the first fc layer
            self.fc0 = nn.Linear(num_fc_input, int(ceil(num_fc_input/2)))
            self.dropout0 = nn.Dropout(1.1*initial_dropout)
            if num_fc_layers == 2:
                # the last fc layer when num_fc_layers == 2
                self.fc1 = nn.Linear(int(ceil(num_fc_input/2)), num_output_nodes)
            else:
                # the middle fc layer
                for i in range(1,num_fc_layers-1):
                    tmp_input = int(ceil(num_fc_input/2**i))
                    tmp_output = int(ceil(num_fc_input/2**(i+1)))
                    exec('self.fc{} = nn.Linear(tmp_input, tmp_output)'.format(i))
                    if i < ceil(num_fc_layers/2) and 1.1**(i+1)*initial_dropout < 1:
                        exec('self.dropout{} = nn.Dropout(1.1**(i+1)*initial_dropout)'.format(i))
                    elif i >= ceil(num_fc_layers/2) and 1.1**(num_fc_layers-1-i)*initial_dropout < 1:
                        exec('self.dropout{} = nn.Dropout(1.1**(num_fc_layers-1-i)*initial_dropout)'.format(i))
                    else:
                        exec('self.dropout{} = nn.Dropout(0.5)'.format(i))
                # the last fc layer
                exec('self.fc{} = nn.Linear(tmp_output, num_output_nodes)'.format(i+1))

        self.to(self.device)
    def forward(self, x):
        if self.num_fc_layers == 1:
            outputs = self.fc0(x)
        else:
            # the first fc layer
            outputs = self.act(self.dropout0(self.fc0(x)))
            if self.num_fc_layers == 2:
                # the last fc layer when num_fc_layers == 2
                outputs = self.fc1(outputs)
            else:
                # the middle fc layer
                for i in range(1,self.num_fc_layers-1):
                    outputs = eval('self.act(self.dropout{}(self.fc{}(outputs)))'.format(i,i))
                # the last fc layer
                outputs = eval('self.fc{}(outputs)'.format(i+1))
        return outputs.softmax(dim=1)
        
class AutoTransformer(nn.Module):
    """
    Class description: Transformer accepts both 1d features and 2d features
    
    """
    def __init__(self, params):
                # **kwargs x1d x2d&dim_2d_input
        super().__init__()
        query = 'trsf_hidden_each_head, head, trsf_num_layer, trsf_dropout, fc_dropout, num_output_nodes, num_fc_layers'
        trsf_hidden_each_head, head, trsf_num_layer, trsf_dropout, fc_dropout, num_output_nodes, num_fc_layers = assign_vars(query, params)
        len_1, len_2, trsf_hidden_dim = 0, 0, trsf_hidden_each_head*head
        self.patience, self.lr = params['patience'], params['learning_rate']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        #--- Feature fusion block ---#
        if 'flag_1d' in params.keys():
            self.input2model_1d = nn.Linear(1, trsf_hidden_dim)
            nn.init.xavier_uniform_(self.input2model_1d.weight)
            len_1 = params['len_1d']
        if 'flag_2d' in params.keys():
            self.input2model_2d = nn.Linear(params['dim_2d_input'], trsf_hidden_dim)
            nn.init.xavier_uniform_(self.input2model_2d.weight)
            len_2 = params['len_2d']
        #--- Transformer ---#        
        self.attention_model = AdaptiveAttentionModel(d_model=trsf_hidden_dim, d_ff=2*trsf_hidden_dim, head=head, 
                                              num_layer=trsf_num_layer, dropout=trsf_dropout)  
        #--- MLP block ---#
        self.to_out = AdaptiveMLP(len_1+len_2, num_output_nodes, num_fc_layers, fc_dropout)  
        #--- Initialize ---#
        for p in self.attention_model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.to_out.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.to(self.device)

    #----- Forward -----#
    def forward(self, x_1d=None, x_2d=None):
        #--- Feature fusion block ---#
        if x_1d is not None and x_2d is not None:
            x_1d = add_dimension(x_1d)
            x_2d = add_dimension(x_2d)
            x_1d = self.input2model_1d(x_1d) # b, l, dim
            x_2d = self.input2model_2d(x_2d)
            x = torch.cat((x_1d, x_2d), 1)
        elif x_1d is not None:
            x_1d = add_dimension(x_1d)
            x = self.input2model_1d(x_1d)
        elif x_2d is not None:
            x_2d = add_dimension(x_2d)
            x = self.input2model_2d(x_2d)
        #--- Transformer block ---#
        x = torch.squeeze(self.attention_model(x), dim=-1)
        #--- MLP block ---#
        y = self.to_out(x)
        del x, x_1d, x_2d
        torch.cuda.empty_cache()
        gc.collect()
        return y

    def to_device(self, x, device):
        if x == []:
            x = None
        else:
            x = x.float().to(device)
        return x
        
    def train_epoch(self, train_iter, optimizer, l, device):
        
        model = self
        model.train()
        train_loss, y_hat_all, y_all = 0, [], []
        for batch_idx, (x_1d, x_2d, y) in enumerate(train_iter):
            x_1d, x_2d, y = self.to_device(x_1d, device), self.to_device(x_2d, device), y.to(device)
            #--- Prediction ---#
            optimizer.zero_grad()
            y_hat = model(x_1d, x_2d)

            loss = l(y_hat, y)
            train_loss += loss.sum()
            loss.sum().backward()
            optimizer.step()

            if device == 'cpu':
                    y_hat_all.extend(y_hat.numpy().tolist())
                    y_all.extend(y.numpy().tolist())
            else:
                y_hat_all.extend(y_hat.detach().cpu().numpy().tolist())
                y_all.extend(y.detach().cpu().numpy().tolist())
            del loss, x_1d, x_2d, y, y_hat
            torch.cuda.empty_cache()
            gc.collect()
        
        train_loss /= len(y_all)
        del model, train_iter, optimizer, l, device
        torch.cuda.empty_cache()
        gc.collect()
        return np.array(y_hat_all), np.array(y_all), train_loss
    
    
    def train_val(self, train_iter, val_iter, test_iter=None):
        device = self.device
        patience, num_epochs, lr = self.patience, 100, self.lr
        early_stopping = EarlyStopping(patience, verbose=False)
        
        optimizer = optim.Adam(self.parameters(), lr=lr)
        #-----  Loss -----#
        loss = nn.CrossEntropyLoss(reduction="none")
        loss = loss.to(device)

        for epoch in range(num_epochs):
            # print('epoch ', epoch)
            _, _, train_loss = self.train_epoch(train_iter, optimizer, loss, device)
            y_hat_val, y_true_val, val_loss = self.test(val_iter)
            if test_iter is not None:
                y_hat_test, y_true_test, test_loss = self.test(test_iter)
            else:
                y_hat_test, y_true_test, test_loss = 0,0,0
            #--- early stop ---#
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        del train_iter, val_iter, optimizer, loss, device
        torch.cuda.empty_cache()
        gc.collect()

        print('Training loss: {:.4f}, val loss: {:f}, test loss: {:f}'.format(train_loss, val_loss, test_loss), flush=True)
        return y_hat_val, y_true_val, y_hat_test, y_true_test
    
    def test(self, test_iter):
        model, device = self, self.device
        model.eval()
        #-----  Loss -----#
        l = nn.CrossEntropyLoss(reduction="none")
        l = l.to(device)
        test_loss, y_hat_all, y_all = 0, [], []
        with torch.no_grad():
            for batch_idx, (x_1d, x_2d, y) in enumerate(test_iter):
                x_1d, x_2d, y = self.to_device(x_1d, device), self.to_device(x_2d, device), y.to(device)
                #--- Prediction ---#
                y_hat = model(x_1d, x_2d)
                loss = l(y_hat, y)
                test_loss += loss.sum()

                if device == 'cpu':
                    y_hat_all.extend(y_hat.numpy().tolist())
                    y_all.extend(y.numpy().tolist())
                else:
                    y_hat_all.extend(y_hat.detach().cpu().numpy().tolist())
                    y_all.extend(y.detach().cpu().numpy().tolist())
                del loss, x_1d, x_2d, y, y_hat
                torch.cuda.empty_cache()
                gc.collect()
            
            test_loss /= len(y_all)
        del model, test_iter, l
        torch.cuda.empty_cache()
        gc.collect()
        return np.array(y_hat_all), np.array(y_all), test_loss

class AutoModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def train_epoch(self, train_iter, optimizer, l, device):
        model = self
        model.train()
        train_loss, y_hat_all, y_all = 0, [], []
        for batch_idx, (x_1d, x_2d, y) in enumerate(train_iter):
            if x_2d == []:
                x_2d = None
            else:
                x_2d = x_2d.float().to(device)
            x_1d, y = x_1d.float().to(device), y.to(device)
            #--- Prediction ---#
            optimizer.zero_grad()
            y_hat = model(x_1d, x_2d)

            loss = l(y_hat, y)
            train_loss += loss.sum()
            loss.sum().backward()
            optimizer.step()

            if device == 'cpu':
                    y_hat_all.extend(y_hat.numpy().tolist())
                    y_all.extend(y.numpy().tolist())
            else:
                y_hat_all.extend(y_hat.detach().cpu().numpy().tolist())
                y_all.extend(y.detach().cpu().numpy().tolist())
        
        train_loss /= len(y_all)
        return np.array(y_hat_all), np.array(y_all), train_loss
    
    
    def train_val(self, train_iter, val_iter, test_iter=None):
        device = self.device
        patience, num_epochs, lr = self.patience, 100, self.lr
        early_stopping = EarlyStopping(patience, verbose=False)
        
        optimizer = optim.Adam(self.parameters(), lr=lr)
        #-----  Loss -----#
        loss = nn.CrossEntropyLoss(reduction="none")
        loss = loss.to(device)

        for epoch in range(num_epochs):
            torch.cuda.empty_cache()
            gc.collect()
            y_hat_train, y_true_train, train_loss = self.train_epoch(train_iter, optimizer, loss, device)
            y_hat_val, y_true_val, val_loss = self.test(val_iter)
            if test_iter is not None:
                y_hat_test, y_true_test, test_loss = self.test(test_iter)
            else:
                y_hat_test, y_true_test, test_loss = 0,0,0
            #--- early stop ---#
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        print('Training loss: {:.4f}, val loss: {:f}, test loss: {:f}'.format(train_loss, val_loss, test_loss), flush=True)

        return y_hat_val, y_true_val, y_hat_test, y_true_test
    
    def test(self, test_iter):
        model, device = self, self.device
        model.eval()
        #-----  Loss -----#
        l = nn.CrossEntropyLoss(reduction="none")
        l = l.to(device)
        test_loss, y_hat_all, y_all = 0, [], []
        with torch.no_grad():
            for batch_idx, (x_1d, x_2d, y) in enumerate(test_iter):
                if x_2d == []:
                    x_2d = None
                else:
                    x_2d = x_2d.float().to(device)
                x_1d, y = x_1d.float().to(device), y.to(device)
                # x_1d, x_2d, y = x_1d.float().to(device), x_2d.float().to(device), y.to(device)
                #--- Prediction ---#
                y_hat = model(x_1d, x_2d)
                loss = l(y_hat, y)
                test_loss += loss.sum()

                if device == 'cpu':
                    y_hat_all.extend(y_hat.numpy().tolist())
                    y_all.extend(y.numpy().tolist())
                else:
                    y_hat_all.extend(y_hat.detach().cpu().numpy().tolist())
                    y_all.extend(y.detach().cpu().numpy().tolist())
            
            test_loss /= len(y_all)
        return np.array(y_hat_all), np.array(y_all), test_loss


class AutoMLP(AutoModel):
    def __init__(self, params):
        super().__init__()
        query = 'num_output_nodes, num_fc_layers, initial_dropout, act'
        num_output_nodes, num_fc_layers, initial_dropout, act = assign_vars(query, params)
        num_fc_input = params['len_1d']
        self.patience, self.lr = params['patience'], params['learning_rate']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_fc_layers, self.act = num_fc_layers, act
        if num_fc_layers == 1:
            self.fc0 = nn.Linear(num_fc_input, num_output_nodes)
        else:
            # the first fc layer
            self.fc0 = nn.Linear(num_fc_input, int(ceil(num_fc_input/2)))
            self.dropout0 = nn.Dropout(1.1*initial_dropout)
            if num_fc_layers == 2:
                # the last fc layer when num_fc_layers == 2
                self.fc1 = nn.Linear(int(ceil(num_fc_input/2)), num_output_nodes)
            else:
                # the middle fc layer
                for i in range(1,num_fc_layers-1):
                    tmp_input = int(ceil(num_fc_input/2**i))
                    tmp_output = int(ceil(num_fc_input/2**(i+1)))
                    exec('self.fc{} = nn.Linear(tmp_input, tmp_output)'.format(i))
                    if i < ceil(num_fc_layers/2) and 1.1**(i+1)*initial_dropout < 1:
                        exec('self.dropout{} = nn.Dropout(1.1**(i+1)*initial_dropout)'.format(i))
                    elif i >= ceil(num_fc_layers/2) and 1.1**(num_fc_layers-1-i)*initial_dropout < 1:
                        exec('self.dropout{} = nn.Dropout(1.1**(num_fc_layers-1-i)*initial_dropout)'.format(i))
                    else:
                        exec('self.dropout{} = nn.Dropout(0.5)'.format(i))
                # the last fc layer
                exec('self.fc{} = nn.Linear(tmp_output, num_output_nodes)'.format(i+1))
        self.to(self.device)

    def forward(self, x, x2=None):
        if self.num_fc_layers == 1:
            outputs = self.fc0(x)
        else:
            # the first fc layer
            outputs = self.act(self.dropout0(self.fc0(x)))
            if self.num_fc_layers == 2:
                # the last fc layer when num_fc_layers == 2
                outputs = self.fc1(outputs)
            else:
                # the middle fc layer
                for i in range(1,self.num_fc_layers-1):
                    outputs = eval('self.act(self.dropout{}(self.fc{}(outputs)))'.format(i,i))
                # the last fc layer
                outputs = eval('self.fc{}(outputs)'.format(i+1))
        return outputs.softmax(dim=1)
###############################################
#---------------   Model tools   -------------#
###############################################

def assign_vars(query, params):
    query = [i.strip() for i in query.split(',')]
    for i in query:
        yield params[i]

def add_dimension(tensor,dim=3):
    shape = tensor.shape
    if dim-len(shape) >= 2:
        tensor = tensor.unsqueeze(0)
        tensor = tensor.unsqueeze(-1)
    elif dim-len(shape) >= 1:
        tensor = tensor.unsqueeze(-1)
    return tensor

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    elif type(m) == nn.LSTM:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])
    elif type(m) == nn.BatchNorm1d:
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    
    """

    def __init__(self, patience=10, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_min = np.Inf
        self.delta = delta

    def __call__(self, val_score, model=None):

        score = -val_score

        if self.best_score is None:
            self.best_score = score
            if model is not None:
                self.save_checkpoint(val_score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if model is not None:
                self.save_checkpoint(val_score, model)
            self.counter = 0

    def save_checkpoint(self, val_score, model):
        '''
        Saves model when validation loss decrease.
        
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_score_min:.6f} --> {val_score:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')     # the parameters of the best model so far
        torch.save(model, 'finish_model.pkl')                 # the best model so far
        self.val_score_min = val_score

def evaluate(y_preds_score, y_trues, average = 'binary', flag_soft=False, verbose=True):
    #average: ‘micro’, ‘macro’, ‘samples’, ‘weighted’, ‘binary’
    if flag_soft==True:
        y_preds_score = softmax(y_preds_score)
        
    bin_enc = preprocessing.OneHotEncoder(sparse=False).fit(y_trues.reshape(-1,1))
    y_preds = y_preds_score.argmax(axis=1)
    # calculate pr-auc
    precision, recall, _ = precision_recall_curve(y_trues, y_preds_score[:,1])
    pr_auc = auc(recall, precision)

    accuracy = accuracy_score(y_trues, y_preds) 
    precision = precision_score(y_trues, y_preds, average=average)
    recall = recall_score(y_trues, y_preds, average=average)
    f1 = f1_score(y_trues, y_preds, average=average)
    y_trues_bin = bin_enc.transform(y_trues.reshape(-1,1))
    roc_auc = roc_auc_score(y_trues_bin, y_preds_score)

    if verbose == True:
        print('accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f} roc_auc:{:.4f} pr_auc:{:.4f}'\
            .format(accuracy, precision, recall, f1, roc_auc, pr_auc))
        print('-'*40)
    return accuracy, precision, recall, f1, roc_auc, pr_auc

def repeat_evaluate(test_accuracy, test_precision, test_recall, test_f1, test_roc_auc, test_pr_auc, y_hat_test, y_test, average='binary', flag_soft=False):
    accuracy, precision, recall, f1, roc_auc, pr_auc= evaluate(y_hat_test, y_test, average=average, verbose=False, flag_soft=flag_soft)
    test_accuracy = test_accuracy + accuracy
    test_precision = test_precision + precision
    test_recall = test_recall + recall
    test_f1 = test_f1 + f1
    test_roc_auc = test_roc_auc + roc_auc
    test_pr_auc = test_pr_auc + pr_auc
    
    return test_accuracy, test_precision, test_recall, test_f1, test_roc_auc, test_pr_auc
###############################################
#----------   Bayesian optimization   --------#
###############################################
def fs_dataset(x_1d, x_2d, f_1d=None, f_2d_0=None, f_2d_1=None):
    idx = pd.IndexSlice
    if (f_1d is not None)&(f_2d_0 is not None)&(f_2d_1 is not None):
        return x_1d[f_1d], x_2d.loc[idx[:,f_2d_0],f_2d_1]
    elif (f_1d is not None)&(f_2d_0 is not None)&(f_2d_1 is None):
        return x_1d[f_1d], x_2d.loc[idx[:,f_2d_0],:]
    elif (f_1d is not None)&(f_2d_0 is None)&(f_2d_1 is not None):
        return x_1d[f_1d], x_2d.loc[:,f_2d_1]
    elif (f_1d is None)&(f_2d_0 is not None)&(f_2d_1 is not None):
        return x_1d, x_2d.loc[idx[:,f_2d_0],f_2d_1]
    elif f_1d is not None:
        return x_1d[f_1d], x_2d
    elif f_2d_0 is not None:
        return x_1d, x_2d.loc[idx[:,f_2d_0],:]
    elif f_2d_1 is not None:
        return x_1d, x_2d.loc[:,f_2d_1]
    else:
        return x_1d, x_2d

def ratio_feature(selected, origin):
    if selected is None or origin is None:
        ratio = 0
    else:
        ratio = len(selected)/len(origin)
    return ratio
def len_2d_feature(val_loader):
    _,tmp_2d,_ = next(iter(val_loader))
    if tmp_2d==[]:
        len_2d_0, len_2d_1 = 0, 0
    elif tmp_2d.dim() == 3:
        len_2d_0, len_2d_1 = tmp_2d.shape[1], tmp_2d.shape[2]
    elif tmp_2d.dim() == 2:
        len_2d_0, len_2d_1 = tmp_2d.shape[0], tmp_2d.shape[1]
    else:
        raise ValueError("tmp_2d must have either 2 or 3 dimensions")
    return len_2d_0, len_2d_1

def auto_nnd(hyperparameter_spaces, n_calls, model_class, determin_hyp, x_1d=None, x_2d=None, y=None, 
batch_size=16, num_outer_split=5, frac_inner_val=0.25, max_len_2d=84, **kwargs):
    
    #---   Decoding hyperparameter search spaces   ---#
    search_spaces = hyperparameter_spaces
    @use_named_args(search_spaces)
    def objective_function(**params):
        x_1d_fs, x_2d_fs = fs_dataset(x_1d, x_2d)
        #--- optimized model hyperparameters ---#
        model_params = {key: value for key, value in params.items() if not key.startswith('feature')}
        model_params.update(determin_hyp)
        print('-'*40)
        print('Model hyperparameters:')
        print('-'*40)
        for key, value in model_params.items():
            print(key, ":", value)


        test_accuracy, test_precision, test_recall, test_f1, test_roc_auc, test_pr_auc = 0, 0, 0, 0, 0, 0
        val_accuracy, val_precision, val_recall, val_f1, val_roc_auc,val_pr_auc = 0, 0, 0, 0, 0, 0
        KF = StratifiedKFold(num_outer_split, shuffle=True)
        for train_index, test_index in KF.split(np.zeros(len(y)),y):
            tmp_train_index, tmp_val_index = next(StratifiedShuffleSplit(n_splits=1, test_size=frac_inner_val, random_state=seed).split(np.zeros(len(y.iloc[train_index])),y.iloc[train_index]))
            train_index, val_index = train_index[tmp_train_index], train_index[tmp_val_index]

            ros = RandomOverSampler(sampling_strategy='auto', random_state=seed)
            train_index, _ = ros.fit_resample(train_index.reshape(-1, 1), y.iloc[train_index])
            train_index = train_index.flatten()
            dataset = FeatureSelectionDataset(x_1d_fs, x_2d_fs, y, max_len_2d)
            train_dataset = Subset(dataset, train_index)
            train_loader = DataLoader(train_dataset, batch_size=batch_size)
            val_dataset = Subset(dataset, val_index)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_dataset = Subset(dataset, test_index)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            tmp_1d,_,_ = next(iter(val_loader))
            len_2d_0, len_2d_1 = len_2d_feature(val_loader)
            #--- construct model with optimized hyperparameters ---#
            model_params.update({'len_1d':tmp_1d.shape[1], 'len_2d':len_2d_0, 'dim_2d_input':len_2d_1})
            model = model_class(model_params)
            model.apply(init_weights)
            y_hat_val, y_true_val, y_hat_test, y_true_test = model.train_val(train_loader, val_loader, test_loader)
           
            val_accuracy, val_precision, val_recall, val_f1, val_roc_auc, val_pr_auc = repeat_evaluate(val_accuracy, val_precision, val_recall, val_f1, val_roc_auc, val_pr_auc, y_hat_val, y_true_val, 'binary', flag_soft=True)
            test_accuracy, test_precision, test_recall, test_f1, test_roc_auc, test_pr_auc = repeat_evaluate(test_accuracy, test_precision, test_recall, test_f1, test_roc_auc, test_pr_auc, y_hat_test, y_true_test, 'binary', flag_soft=True)

            i = num_outer_split
        print('-'*40)
        print('Overall performance on validation set:')
        print('accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f} auc:{:.4f}'\
            .format(val_accuracy/i, val_precision/i, val_recall/i, val_f1/i, val_roc_auc/i))
        print('Overall performance on test set:')
        print('accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f} auc:{:.4f}'\
            .format(test_accuracy/i, test_precision/i, test_recall/i, test_f1/i, test_roc_auc/i))
        print('-'*40)
        
        return -0.25*(val_accuracy+test_accuracy)/num_outer_split-0.25*(val_roc_auc+test_roc_auc)/num_outer_split
      
    gp = gp_minimize(objective_function, search_spaces, n_calls=n_calls, random_state=seed, acq_func="EI",verbose=True, n_jobs=-1)
    print('*'*40)
    print('return -0.25*(val_accuracy+test_accuracy)/num_outer_split-0.25*(val_roc_auc+test_roc_auc)/num_outer_split')
    print('*'*40)
    return gp, search_spaces

def auto_hpo_nnd_fs(hyperparameter_spaces, n_calls, model_class, determin_hyp, 
x_1d_feature=None, x_2d_feature_0=None, x_2d_feature_1=None, x_1d=None, x_2d=None, y=None, 
batch_size=16, feature_batch=10, num_outer_split=5, frac_inner_val=0.25, max_len_2d=84, **kwargs):
    
    #---   Decoding hyperparameter search spaces   ---#
    feature_search_spaces, feature_search_spaces_len = construct_feature_search_space(x_1d_feature, x_2d_feature_0, x_2d_feature_1, feature_batch)
    search_spaces = feature_search_spaces + hyperparameter_spaces
    @use_named_args(search_spaces)
    def objective_function(**params):
        #--- selected feature ---#
        feature_params = {key: value for key, value in params.items() if key.startswith('feature')}
        f_1d, f_2d_0, f_2d_1 = int2feature(feature_params, feature_search_spaces_len, x_1d_feature, x_2d_feature_0, x_2d_feature_1)
        print('-'*40)
        print('Selected feature:')
        print('-'*40)
        print('(1d){}\n, (2d_0){}\n, (2d_1){}'.format(f_1d, f_2d_0, f_2d_1))
        ratio_1d, ratio_2d_1 = ratio_feature(f_1d, x_1d_feature), ratio_feature(f_2d_1, x_2d_feature_1)
        x_1d_fs, x_2d_fs = fs_dataset(x_1d, x_2d, f_1d, f_2d_0, f_2d_1)
        #--- optimized model hyperparameters ---#
        model_params = {key: value for key, value in params.items() if not key.startswith('feature')}
        model_params.update(determin_hyp)
        print('-'*40)
        print('Model hyperparameters:')
        print('-'*40)
        for key, value in model_params.items():
            print(key, ":", value)
        test_accuracy, test_precision, test_recall, test_f1, test_roc_auc, test_pr_auc = 0, 0, 0, 0, 0, 0
        val_accuracy, val_precision, val_recall, val_f1, val_roc_auc,val_pr_auc = 0, 0, 0, 0, 0, 0
        KF = StratifiedKFold(num_outer_split, shuffle=True)
        for train_index, test_index in KF.split(np.zeros(len(y)),y):
            tmp_train_index, tmp_val_index = next(StratifiedShuffleSplit(n_splits=1, test_size=frac_inner_val, random_state=seed).split(np.zeros(len(y.iloc[train_index])),y.iloc[train_index]))
            train_index, val_index = train_index[tmp_train_index], train_index[tmp_val_index]

            ros = RandomOverSampler(sampling_strategy='auto', random_state=seed)
            train_index, _ = ros.fit_resample(train_index.reshape(-1, 1), y.iloc[train_index])
            train_index = train_index.flatten()
            dataset = FeatureSelectionDataset(x_1d_fs, x_2d_fs, y, max_len_2d)
            train_dataset = Subset(dataset, train_index)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4)
            val_dataset = Subset(dataset, val_index)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)
            test_dataset = Subset(dataset, test_index)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

            #--- construct model with optimized hyperparameters ---#
            torch.cuda.empty_cache()
            gc.collect()
            len_2d_0, len_2d_1 = len_2d_feature(val_loader)
            len_1d = 0 if f_1d is None else len(f_1d)
            model_params.update({'len_1d':len_1d, 'len_2d':len_2d_0, 'dim_2d_input':len_2d_1})
            model = model_class(model_params)
            model.apply(init_weights)
            y_hat_val, y_true_val, y_hat_test, y_true_test = model.train_val(train_loader, val_loader, test_loader)
            if torch.cuda.is_available():
                model = model.cpu()
                torch.cuda.empty_cache()
            del train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader, model
            torch.cuda.empty_cache()
            gc.collect()
           
            val_accuracy, val_precision, val_recall, val_f1, val_roc_auc, val_pr_auc = repeat_evaluate(val_accuracy, val_precision, val_recall, val_f1, val_roc_auc, val_pr_auc, y_hat_val, y_true_val, 'binary', flag_soft=True)
            test_accuracy, test_precision, test_recall, test_f1, test_roc_auc, test_pr_auc = repeat_evaluate(test_accuracy, test_precision, test_recall, test_f1, test_roc_auc, test_pr_auc, y_hat_test, y_true_test, 'binary', flag_soft=True)
            i = num_outer_split
        print('-'*40)
        print('Overall performance on validation set:')
        print('accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f} auc:{:.4f}'\
            .format(val_accuracy/i, val_precision/i, val_recall/i, val_f1/i, val_roc_auc/i))
        print('Overall performance on test set:')
        print('accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f} auc:{:.4f}'\
            .format(test_accuracy/i, test_precision/i, test_recall/i, test_f1/i, test_roc_auc/i))
        print('-'*40)

        return -0.25*(val_accuracy+test_accuracy)/num_outer_split-0.25*(val_roc_auc+test_roc_auc)/num_outer_split+0.015*ratio_1d+0.015*ratio_2d_1

    gp = gp_minimize(objective_function, search_spaces, n_calls=n_calls, random_state=seed, acq_func="EI",verbose=True, n_jobs=-1)
    print('*'*40)
    print('return -0.25*(val_accuracy+test_accuracy)/num_outer_split-0.25*(val_roc_auc+test_roc_auc)/num_outer_split+0.015*ratio_1d+0.015*ratio_2d_1')
    print('*'*40)
    return gp, search_spaces, feature_search_spaces_len

###############################################
#------------   Optimization tools   ---------#
###############################################
def split_search_space(x, feature_batch, feature_flag='1d'):
    
    num_group = len(x)//feature_batch
    mod_feature_batch = len(x)%feature_batch
    params, params_len = [], dict()

    for i in range(num_group):
        key_name = 'feature_' + feature_flag +'_group_' + str(i)
        params.append(Integer(1, 2**feature_batch-1, name=key_name))
        params_len[key_name] =  feature_batch
        
    if mod_feature_batch != 0:
        if num_group == 0:
            i = -1
        key_name = 'feature_' + feature_flag +'_group_' + str(i+1)
        params.append(Integer(1,2**mod_feature_batch-1, name=key_name))
        params_len[key_name] =  mod_feature_batch
        
    return params, params_len

def construct_feature_search_space(x_1d_feature=None, x_2d_feature_0=None, x_2d_feature_1=None, feature_batch = 20):

    search_spaces, search_spaces_len = [], dict()

    if x_2d_feature_0 is not None:
        params, params_len = split_search_space(x_2d_feature_0, feature_batch, '2d_0')
        search_spaces.extend(params)
        search_spaces_len.update(params_len)

    if x_2d_feature_1 is not None:
        params, params_len = split_search_space(x_2d_feature_1, feature_batch, '2d_1')
        search_spaces.extend(params)
        search_spaces_len.update(params_len)

    if x_1d_feature is not None:
        params, params_len = split_search_space(x_1d_feature, feature_batch, '1d')
        search_spaces.extend(params)
        search_spaces_len.update(params_len)   

    return search_spaces, search_spaces_len

def int2bin_group(feature_params_value, feature_search_spaces_len_value):
    bin_selected_features = bin(feature_params_value)[2:]
    for i in range(feature_search_spaces_len_value-len(bin_selected_features)):
        bin_selected_features = '0' + bin_selected_features
    return bin_selected_features

def int2bin(bin_selected_features, feature_params_value, feature_search_spaces_len_value):
    if bin_selected_features is None:
        bin_selected_features = int2bin_group(feature_params_value, feature_search_spaces_len_value)
    else:
        bin_selected_features = bin_selected_features + int2bin_group(feature_params_value, feature_search_spaces_len_value)
    return bin_selected_features

def bin2feature(feature_to_select, bin_selected_features):
    if (feature_to_select is not None) & (bin_selected_features is not None):
        return [feature_to_select[i] for i in range(len(feature_to_select)) if bin_selected_features[i] == '1']
    else:
        return None

def int2feature(feature_params, feature_search_spaces_len, x_1d_feature=None, x_2d_feature_0=None, x_2d_feature_1=None):
    bin_selected_features_1d, bin_selected_features_2d_0, bin_selected_features_2d_1 = None, None, None
    for i in feature_params.keys():
        if i.startswith('feature_1d'):
            bin_selected_features_1d = int2bin(bin_selected_features_1d, feature_params[i], feature_search_spaces_len[i])
        elif i.startswith('feature_2d_0'):
            bin_selected_features_2d_0 = int2bin(bin_selected_features_2d_0, feature_params[i], feature_search_spaces_len[i])
        elif i.startswith('feature_2d_1'):
            bin_selected_features_2d_1 = int2bin(bin_selected_features_2d_1, feature_params[i], feature_search_spaces_len[i])
    f_1d = bin2feature(x_1d_feature, bin_selected_features_1d)
    f_2d_0 = bin2feature(x_2d_feature_0, bin_selected_features_2d_0)
    f_2d_1 = bin2feature(x_2d_feature_1, bin_selected_features_2d_1)
    return f_1d, f_2d_0, f_2d_1

###############################################
#---------   Obtain attention tools   --------#
###############################################
def attention_score(patientID, x_1d_fs, x_2d_fs, model):
    feature_list = ['num_snv']
    feature_list.extend(x_2d_fs.loc[patientID].index)
    num_snv = int(x_1d_fs.loc[patientID].values.item()+1)

    x1 = torch.tensor(x_1d_fs.loc[patientID].values, dtype=torch.float64)

    x2 = TruncatePadding(flatten2array(x_2d_fs.loc[patientID]), 84)
    x2 = torch.tensor(x2, dtype=torch.float64)

    x1 = x1.float().cuda()
    x2 = x2.float().cuda()
    x2 = torch.unsqueeze(x2, dim=0)
    model.attention_model.attn_matrices=[]
    model(x1, x2)

    attn_matrix = torch.squeeze(model.attention_model.attn_matrices[0],dim=0).cpu().detach().numpy()
    attn_matrix = attn_matrix[:,:num_snv,:num_snv]
    # normalize
    attn_matrix = (attn_matrix - attn_matrix.min(axis=(1, 2), keepdims=True)) / (
    attn_matrix.max(axis=(1, 2), keepdims=True) - attn_matrix.min(axis=(1, 2), keepdims=True))
    # average accross head
    avg_attn_matrix = pd.DataFrame(np.mean(attn_matrix, axis=0), index=feature_list, columns=feature_list)
    attn_head1 = pd.DataFrame(attn_matrix[0], index=feature_list, columns=feature_list)
    attn_head2 = pd.DataFrame(attn_matrix[1], index=feature_list, columns=feature_list)
    attn_head3 = pd.DataFrame(attn_matrix[2], index=feature_list, columns=feature_list)
    attn_head4 = pd.DataFrame(attn_matrix[3], index=feature_list, columns=feature_list)

    return [avg_attn_matrix, attn_head1, attn_head2, attn_head3, attn_head4]

def union_attention(a, b):
    """
    a, b: pandas.DataFrame
    """
    unique_gene = ~b.index.duplicated(keep='first')
    b = b.loc[unique_gene, unique_gene]

    union_features = list(set(a.columns).union(b.columns))

    union_attention_matrix = pd.DataFrame(0, index=union_features, columns=union_features)

    for df in [a, b]:
        for i in df.columns:
            for j in df.columns:
                union_attention_matrix.at[i, j] += df.at[i, j]

    return union_attention_matrix

def union_attention_list(df_list):
    """
    df_list: List[pandas.DataFrame]
    """
    df_list = [df.loc[~df.index.duplicated(keep='first'), ~df.columns.duplicated(keep='first')] for df in df_list]
    union_index = df_list[0].index
    union_columns = df_list[0].columns
    for df in df_list[1:]:
        union_index = union_index.union(df.index)
        union_columns = union_columns.union(df.columns)
    df_list = [df.reindex(index=union_index, columns=union_columns, fill_value=0) for df in df_list]
    result = pd.concat(df_list).groupby(level=0).sum()
    return result

def obtain_stage_attention(x_stage, base_path):
    attn_head1, attn_head4 = [], []
    for patientID in x_stage.index:
        path = base_path+patientID+'_attention.pickle'
        with open(path, 'rb') as f:
            attn_matrix = pickle.load(f)
        attn_head1.append(attn_matrix[1])
        attn_head4.append(attn_matrix[4])
    attn_head1 = union_attention_list(attn_head1)
    attn_head4 = union_attention_list(attn_head4)
    attn_matrix = attn_head1.add(attn_head4)
    return attn_matrix

###############################################
#---------   Obtain top gene tools   ---------#
###############################################
def top_gene_att_all(attn_matrix, num_gene=20):
    np.fill_diagonal(attn_matrix.values, 0)
    col_sums = attn_matrix.sum()
    top_gene = col_sums.nlargest(num_gene+1)
    if 'num_snv' in top_gene.index:
        top_gene = top_gene[1:]
    else:
        top_gene = top_gene[:num_gene]
    return top_gene    

###############################################
#-------   Obtain top gene pair tools   ------#
###############################################
def norm_tri(df):
    lower_triangle_indices = np.tril_indices(df.shape[0], k=-1)
    lower_triangle_data = df.values[lower_triangle_indices]
    normalized_lower_triangle_data = (lower_triangle_data - lower_triangle_data.min()) / (lower_triangle_data.max() - lower_triangle_data.min())
    df_normalized = df.copy()
    df_normalized.values[lower_triangle_indices] = normalized_lower_triangle_data
    return df_normalized

def tri_attention(attn_matrix):
    attn_matrix = attn_matrix.drop(columns=['num_snv'])
    attn_matrix = attn_matrix.drop(index=['num_snv'])
    np.fill_diagonal(attn_matrix.values, 0)
    df_diag_sum = attn_matrix.add(attn_matrix.T)
    df_diag_sum = norm_tri(df_diag_sum)
    return df_diag_sum

def top_gene_pair_att(df_diag_sum, num=50):
    long_format = df_diag_sum.unstack().reset_index()
    long_format.columns = ['Gene1', 'Gene2', 'Weight']
    long_format = long_format[long_format['Gene1'] != long_format['Gene2']]
    sorted_long_format = long_format.sort_values(by='Weight', ascending=False)
    return sorted_long_format[:num*2][::2]


