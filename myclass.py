#!/usr/bin/env python
# coding: utf-8

# # Preprocessing class and function

# In[ ]:

import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import warnings
import time
import copy
import math


import torch.autograd as autograd
import torch.nn.functional as F


from sklearn.metrics import *

from torch import nn
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from IPython import display
from copy import deepcopy
from numpy import *
from random import gauss
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from sklearn.ensemble import AdaBoostClassifier
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
from d2l import torch as d2l

##################################################
#---- Data Preprocessing Function defination ---#
################################################

#----- OneHot categorical variable -----#
def OneHotText(x):
    
    """
    Function description: one hot encoding from text to integer and then one_hot_form
    
    """
    LabelTextEncoder = preprocessing.LabelEncoder()
    one_hot = LabelTextEncoder.fit_transform(x).reshape(len(x),1)
    OneHotTextEncoder = preprocessing.OneHotEncoder(sparse=False)
    one_hot = OneHotTextEncoder.fit_transform(one_hot)
    
    return one_hot,LabelTextEncoder,OneHotTextEncoder

#----- Embedding categorical variable -----#
def EmbeddingSNV(x,max_variats,embedding_dim = 4,feature = 'Gene'):
    """
    Function description: embedding weights return by considering a patient's information as a sentence
    
    """
    num_embeddings =  len(x[feature].value_counts())+1
    # LabelGene = preprocessing.LabelEncoder()
    # snv_gene['Gene'] = LabelGene.fit_transform(snv_gene['Gene'])+1#.reshape(len(x),1)
    x = np.array(x.groupby('patientID'),dtype = object)[:,1]
    sentence = None
    for i in range(len(x)):
        if sentence is None:
            sentence = [PaddingToMax(x[i][feature], max_variats).T]
        else:
            sentence = np.concatenate((sentence, [PaddingToMax(x[i][feature], max_variats).T]), axis=0)

    input = torch.LongTensor(sentence)
    
    SNVEmbedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0, max_norm=1)
    SNVEmbedding(input)
    SNVEmbedding.weight.requires_grad = False
    
    return SNVEmbedding.weight.numpy().tolist()

def PaddingToMax(x,max_variats):
    """
    Function description: truncate or padding variable-length variants
    
    """
    if len(x) >= max_variats:
        return x[:max_variats]  # truncate
    else:
        return np.append(x,np.zeros((max_variats - len(x))), axis=0) 

#----- Normalization -----#
def MinMaxScale(x):
    
    """
    Function description: implement the minimax normalization
    
    """
    minn = np.min(x)
    maxx = np.max(x)
    x = (x-minn)/(maxx-minn)
    return x

def MinMaxLogScale(x):
    
    """
    Function description: compute the logarithm of all numeric features forst and then implement the minimax normalization
    
    """
    if any([i<0 for i in x]):
        x = [i+abs(min(x)) for i in x]
        
    if any([i==0 for i in x]):
        nozero = min(log10(list(filter(lambda val: abs(val)>=  10**-10, x))))-10
        x = [i+10**nozero for i in x]
    x = log10(x)
    minn = np.min(x)
    maxx = np.max(x)
    x = (x-minn)/(maxx-minn)
    return x

class Normalization:
    def __init__(self, method, **kwargs):
        super(Normalization, self).__init__(**kwargs)
        self.method = method
        
    def normalize(self,x):
        if self.method == 'MinMaxScale':
            return MinMaxScale(x)
        if self.method == 'MinMaxLogScale':
            return MinMaxLogScale(x)
        
def normalizer(x, list_to_normalization, scale):
    for i in list_to_normalization:
        x[i] = scale.normalize(x[i])
    return x


#----- Flatten SNV -----#
def FlattenSNV(snv_multiindex):
    
    """
    Function description: extract and flatten all SNV features into a 2D array
    
    """
    snv_array = []
    for i in range(snv_multiindex.shape[0]):
        temp1 = []
        for j in snv_multiindex.iloc[i]:
            if (type(j) == np.ndarray or type(j) == list):
                for k in j:
                    temp1.append(k)
            else:
                temp1.append(j)
        snv_array.append(temp1)
    snv_array = np.array(snv_array) 
    return snv_array


#----- Truncate and padding SNV -----#
def TruncatePadding(snv_array, max_variats):
    
    """
    Function description: truncate or padding variable-length variants
    
    """
    if len(snv_array) >= max_variats:
        return snv_array[:max_variats]  # truncate
    else:
        return np.append(snv_array,np.zeros((max_variats - snv_array.shape[0],snv_array.shape[1])), axis=0) # padding with zero array


# In[ ]:


# new
def MatchSNV(x, snv):
    x_snv = None
    for i in x.index:
        if x_snv is None:
            x_snv = snv[snv.patientID == i]
        else:
            x_snv = x_snv.append(snv[snv.patientID == i])
            
    return x_snv

def GroupAndExtractSNV(x1, x_snv, max_variats):
    x = deepcopy(x1)
    num_patients  = len(x)
    snv_group = np.array(x_snv.groupby('patientID'),dtype = object)
        #----- Flatten SNV -----#
    len_flatten_variants = FlattenSNV(snv_group[0][1].drop(columns='patientID')).shape[1] # length of flatten variation
        #----- Truncate and padding SNV -----#
    x['snv'] = [np.zeros((max_variats,len_flatten_variants)) for i in range(num_patients)] # fill snv with 0
    x['snv_valid_len'] = np.zeros(num_patients)
    for i in range(snv_group.shape[0]):
        patient_id = snv_group[i][0]
    #     print(patient_id)
        snv_multiindex = snv_group[i][1].drop(columns='patientID')
        snv_array =  FlattenSNV(snv_multiindex) # Flatten
        num_repeat = len(x[(x.index == patient_id)])
    #     print(x[(x.index == patient_id)])
        if num_repeat == 1:
            x.loc[patient_id,'snv_valid_len'] = snv_array.shape[0]# Denote valid length
    #         print(patient_id)
    #             print(x.at[patient_id,'snv'])
    #             print(TruncatePadding(snv_array, max_variats))
            x.loc[patient_id,'snv'] = [TruncatePadding(snv_array, max_variats)]# Truncate & padding

        else:
            tmp = x.loc[patient_id].iloc[0]
    #             print(tmp)
            tmp['snv_valid_len'] = snv_array.shape[0]# Denote valid length
            tmp['snv'] = TruncatePadding(snv_array, max_variats)# Truncate & padding
            x = x.drop([patient_id], axis=0)
            for i in range(num_repeat):
                x = x.append(tmp) 
    return x

def NewGroupAndExtractSNV(x1, x_snv, max_variats):
    x = deepcopy(x1)
    num_patients  = len(x)
    snv_group = np.array(x_snv.groupby('patientID'),dtype = object)
        #----- Flatten SNV -----#
    len_flatten_variants = FlattenSNV(snv_group[0][1].drop(columns='patientID')).shape[1] # length of flatten variation
        #----- Truncate and padding SNV -----#
    x['snv'] = [np.zeros((max_variats,len_flatten_variants)) for i in range(num_patients)] # fill snv with 0
    for i in range(snv_group.shape[0]):
        patient_id = snv_group[i][0]
    #     print(patient_id)
        snv_multiindex = snv_group[i][1].drop(columns='patientID')
        snv_array =  FlattenSNV(snv_multiindex) # Flatten
        num_repeat = len(x[(x.index == patient_id)])
    #     print(x[(x.index == patient_id)])
        if num_repeat == 1:
            x.loc[patient_id,'snv'] = [TruncatePadding(snv_array, max_variats)]# Truncate & padding

        else:
            tmp = x.loc[patient_id].iloc[0]
            tmp['snv'] = TruncatePadding(snv_array, max_variats)# Truncate & padding
            x = x.drop([patient_id], axis=0)
            for i in range(num_repeat):
                x = x.append(tmp) 
    return x


def PreprocessSNV(x,max_variats):
    
    x_snv = MatchSNV(x)
            
    return GroupAndExtractSNV(x, x_snv, max_variats)

def DataAugmentation(x,list_to_noise, mu, sigma):
    for i in list_to_noise:
        x[i] += [gauss(mu,sigma)*random.randint(0,2) for i in range(len(x))]
        
    return x

def Resampler(x,y): 
    random.seed(1105)
    lable_counts = y.value_counts()
    list_to_resample = lable_counts.index.drop(lable_counts.index[argmax(lable_counts)])
    for i in list_to_resample:
        num_resample = max(lable_counts) - lable_counts.loc[i]
        resample_index = random.randint(lable_counts.loc[i], size=num_resample)
        x = x.append(x[x.label == i].iloc[resample_index])
    return x

def UnionCNVSNVClinical(sub_train):
    length = sub_train.iloc[0]['snv'].shape[0]*sub_train.iloc[0]['snv'].shape[1]
    snv1 = zeros((len(sub_train),length))
    for i in range(len(sub_train)):
        snv1[i,:] = sub_train.iloc[i]['snv'].to_numpy().reshape(1,length)
    print(snv1.shape)

    sub_train_x = np.concatenate([snv1, sub_train[['cnv_Count_5mb','cnv_Count_gistic','Age (years)', 'Sex', 'Smoker', \
        'Plasma volume used', 'Plasma cfDNA concentration (ng/mL)', 'Plasma DNA input (ng)']]], axis=1)
    sub_train_y =  sub_train['label']
    return sub_train_x, sub_train_y


def EmbeddingSNV(x,max_variats,embedding_dim = 4,feature = 'Gene'):
    """
    Function description: embedding weights return by considering a patient's information as a sentence
    
    """
    x_before = deepcopy(x)
    num_snv = len(x)
    num_embeddings =  len(x[feature].value_counts())+1
    x = np.array(x.groupby('patientID'),dtype = object)[:,1]
    sentence = None
    for i in range(len(x)):
        if sentence is None:
            sentence = [PaddingToMax(x[i][feature], max_variats).T]
        else:
            sentence = np.concatenate((sentence, [PaddingToMax(x[i][feature], max_variats).T]), axis=0)

    input = torch.LongTensor(sentence)
    
    SNVEmbedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0, max_norm=1)
    SNVEmbedding(input)
    SNVEmbedding.weight.requires_grad = False
    SNVEmbedding = SNVEmbedding.weight.numpy().tolist()
    
    embedded_snv = []
    for i in range(num_snv):
        embedded_snv.append(SNVEmbedding[x_before.loc[i,feature]][:])
    return  embedded_snv

###############################################
#-----------   Feature Selection   -----------#
###############################################

def select_gene_snv_feature(len_feature, snv_selected, gene_selected, x):
    x = deepcopy(x)
    x['gene_snv'] = ""
    for i in range(len(x)):
        tmp = deepcopy(x.iloc[i]['snv'].loc[gene_selected][snv_selected])
        x.at[x.index[i],'gene_snv'] = FlattenSNV(tmp)
    x = x.drop('snv', axis = 1).rename(columns={'gene_snv':'snv'})
    return x


# # Model class and function

# In[ ]:


###############################################
 #--------------   Data iterator   ------------#
###############################################

class BatchDataGeneration:
  
  """
  Class description: to generate data for batch
  
  """
  def __init__(self, data, batch_size, shuffle=True, **kwargs):
      super(BatchDataGeneration, self).__init__(**kwargs)
      
      self.data = data
      self.batch_size = batch_size
      self.shuffle = shuffle
      
  def generate(self):
      """
      Function description: randomly split data into different batch and reter iterator of data generation
      
      """
  
      num_splits = ceil(len(self.data)/self.batch_size)
      if self.shuffle:
          data = sklearn.utils.shuffle(self.data)
      data = iter(np.array_split(data, num_splits))
      return data   
  
###############################################
 #---------------   Model Tools   -------------#
###############################################
class Accumulator:
  """
  Class description: summing over n variables
  
  """
  def __init__(self, n):
      self.data = [0.0] * n

  def add(self, *args):
      self.data = [a + float(b) for a, b in zip(self.data, args)]

  def reset(self):
      self.data = [0.0] * len(self.data)

  def __getitem__(self, idx):
      return self.data[idx]
  

def binary_accuracy(preds, y):
  """
  Function description: compute accuracy for binary classification
  
  """
  rounded_preds = torch.round(preds)
  correct = (rounded_preds == y).float() 
  acc = correct.sum() / len(correct)
  return acc


def grad_clipping(net, theta):
    """
    Function description: clip the gradient with large value
  
    """
    params = net.parameters()
    norm = math.sqrt(sum((p.grad ** 2).sum() for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


# In[ ]:
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

    def __call__(self, val_score, model):

        score = -val_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
#             print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
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



###############################################
 #---------------   Model Define   ------------#
###############################################
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
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
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
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
        super(MultiHeadedAttention, self).__init__()
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
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
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

class AttentionModel(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, d_input, d_model, d_ff, head, num_layer, dropout):
        super(AttentionModel, self).__init__()
        # c = copy.deepcopy
        # attn0 = MultiHeadedAttention(head, d_input, d_model)
        attn = MultiHeadedAttention(head, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        # position = PositionalEncoding(d_model, dropout)
        # layer0 = EncoderLayer(d_model, c(attn0), c(ff), dropout)
        layer = EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), dropout)
        self.layers = clones(layer, num_layer)
        # layerlist = [copy.deepcopy(layer0),]
        # for _ in range(num_layer-1):
        #     layerlist.append(copy.deepcopy(layer))
        # self.layers = nn.ModuleList(layerlist)
        self.norm = LayerNorm(layer.size)
        self.posi = PositionalEncoding(d_model, dropout)
        self.input2model = nn.Linear(d_input, d_model)

    def forward(self, x):
        "Pass the input (and mask) through each layer in turn."
        # x: embedding (b,l,we)
        x = self.input2model(x)
        x = self.posi(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)



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

#################################################
 #----   Train and evaluate with early stop  ----#
#################################################
def train_epoch(model, train_iter, loss, optimizer, device, clip_tresh=100):
  
  """
  Fuction description: function to train the model for an epoch
  
  """
  #--- Initialization ---#
  model.train()  # Set to training mode
  timer = d2l.Timer()
  metric = Accumulator(4)  # loss, acc, auc, num of samples
  iter_count = 0

  for data in train_iter.generate():
      #--- Obtain lable and feature from iterator ---#
      y = torch.tensor(data['label'].to_numpy()).reshape(data.shape[0])
      x_snv = []
      for i in range(len(data)):
          x_snv.append(data.iloc[i]['snv'])
      x_snv = torch.tensor(np.array(x_snv), dtype=torch.float32)
      x_other = torch.tensor(data.drop(['snv','label'], axis=1).to_numpy(), dtype=torch.float32)
      x_snv, x_other, y = x_snv.to(device), x_other.to(device),  y.to(device)
    
      #--- Prediction ---#
      y_hat = model(x_snv, x_other)
  
      #--- Compute performance metric ---#
      l = loss(y_hat, y) 
      acc = d2l.accuracy(y_hat, y)
      auc = roc_auc_score(y.detach().cpu().numpy(), y_hat[:,1].detach().cpu().numpy())
      
      #--- Compute performance metric ---#
      optimizer.zero_grad()
      l.sum().backward()
    #   grad_clipping(model, clip_tresh)
      optimizer.step()
      metric.add(l.sum(), acc, auc, y.numel())
      iter_count += 1
      
  return metric[0]/metric[3], metric[1]/metric[3], metric[2]/iter_count, metric[3]/timer.stop()

def evaluate(model, val_iter, loss, device):
  
  """
  Fuction description: function to evaluate the model
  
  """
  #--- Initialization ---#
  model.eval() # stop using dropout
  metric = Accumulator(4)  # loss, acc, auc, num of samples
  iter_count = 0
 
  with torch.no_grad():  # cancle autograd
      for data in val_iter.generate():
          #--- Obtain lable and feature from iterator ---#
          y = torch.tensor(data['label'].to_numpy()).reshape(data.shape[0])
          x_snv = []
          for i in range(len(data)):
              x_snv.append(data.iloc[i]['snv'])
          x_snv = torch.tensor(np.array(x_snv), dtype=torch.float32)
          x_other = torch.tensor(data.drop(['snv','label'], axis=1).to_numpy(), dtype=torch.float32)
          x_snv, x_other, y = x_snv.to(device), x_other.to(device),  y.to(device)
          
          #--- Prediction ---#
          y_hat = model(x_snv, x_other)

          #--- Compute performance metric ---#
          l = loss(y_hat, y) # mean over this batch
          acc = d2l.accuracy(y_hat, y)
          auc = roc_auc_score(y.detach().cpu().numpy(), y_hat[:,1].detach().cpu().numpy())
          metric.add(l.sum(), acc, auc, y.numel())
          iter_count += 1
  return metric[0]/metric[3], metric[1]/metric[3], metric[2]/iter_count

def train_evaluate(model, train_iter, val_iter, loss, optimizer, early_stopping, num_epochs, device):
  
  """
  Fuction description: function to train and evaluate the model
  
  """
#   animator1 = d2l.Animator(xlabel='epoch', ylabel='loss', legend=['train_loss','val_loss'], xlim=[1, num_epochs])
#   animator2 = d2l.Animator(xlabel='epoch', ylabel='Accuracy', legend=['train_acc','val_acc'], xlim=[1, num_epochs])
#   animator3 = d2l.Animator(xlabel='epoch', ylabel='AUC', legend=['train_auc','val_auc'], xlim=[1, num_epochs])
#   animator4 = d2l.Animator(xlabel='epoch', ylabel='Grad', legend=['grad_norm'], xlim=[10, num_epochs])
  
  #--- Training and prediction ---#
  for epoch in range(num_epochs):
      train_loss, train_acc, train_auc, train_speed = train_epoch(model, train_iter, loss, optimizer, device, clip_tresh=100)
      val_loss, val_acc, val_auc = evaluate(model, val_iter, loss, device)
    #   if (epoch + 1) % 1 == 0:
    #       animator1.add(epoch + 1, (train_loss, val_loss))
    #       animator2.add(epoch + 1, (train_acc, val_acc))
    #       animator3.add(epoch + 1, (train_auc, val_auc))
    #       animator4.add(epoch + 1, (math.sqrt(sum((p.grad ** 2).sum() for p in model.parameters()))))
      #--- early stop ---#
      early_stopping(val_loss, model)
      if early_stopping.early_stop:
        #   print("Early stopping")
          break
          
  return train_loss, train_acc, train_auc, train_speed, val_loss, val_acc, val_auc

