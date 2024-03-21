from utils import *
from copy import deepcopy
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

import random
import skopt
import argparse
import os
import gc
import numpy as np
import pandas as pd


from scipy.stats import pearsonr
from sklearn.feature_selection import VarianceThreshold

import warnings
warnings.filterwarnings("ignore")

import datetime
import time
    
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
        patience, num_epochs, lr = self.patience, 50, self.lr
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
                if x_2d == []:
                    x_2d = None
                else:
                    x_2d = x_2d.float().to(device)
                x_1d, y = x_1d.float().to(device), y.to(device)
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


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1105,
                        help='random seed (default: 1105)')
    parser.add_argument('--max_len_2d', type=int, default=0,
                        help='max length of 2d feature like SNVs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size (default: 256)')
    parser.add_argument('--num_outer_split', type=int, default=5,
                        help='parameter for data splitting')
    parser.add_argument('--frac_inner_val', type=float, default=0.25,
                        help='parameter for data splitting')
    parser.add_argument('--feature_batch', type=int, default=20,
                        help='the maximum value of a feature group for FS (default: 10)')
    parser.add_argument('--n_calls', type=int, default=50,
                        help='the maximum evolution step for Bayesian optimization (default: 100)')
    parser.add_argument('--start_repeat', type=int, default=0,
                        help='the number of repeats')
    parser.add_argument('--num_repeats', type=int, default=10,
                        help='the number of repeats')
    parser.add_argument('--x_1d_train_path', default='./dataset/LUCAS_cohort/frag_mutation_clinic.csv',
                        help='path for loading the 1D training data including label')
    args = parser.parse_args()

    #################################################
    #---------------- Seed setup ----------------#
    #################################################
    

    def remove_highly_correlated_features(df, threshold=0.95):
        corr_matrix = df.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        return df.drop(to_drop, axis=1)

    def remove_near_zero_variance_features(df, threshold=0.01):
        selector = VarianceThreshold(threshold)
        selector.fit(df)
        return df[df.columns[selector.get_support(indices=True)]]


    #################################################
    #------------------ Load Data ------------------#
    #################################################
    script_directory = os.path.dirname(os.path.abspath(__file__))
    path = script_directory+'/../result/comparison_result/lucas/AutoCancer.csv'

    x_1d = pd.read_csv(script_directory+'/../'+args.x_1d_train_path, index_col=0)
    x_1d.set_index('id', inplace=True)
    
    x_1d['Type'] = x_1d['Type'].replace({'healthy': 0, 'cancer': 1})
    x_1d['Gender'] = x_1d['Gender'].replace({'Female': 0, 'Male': 1})

    scaler = StandardScaler()
    x_1d['Age'] = scaler.fit_transform(x_1d['Age'].values.reshape(-1, 1))
    x_1d = x_1d.rename(columns={'Type': 'label'})

    zscore_ratio_columns = [col for col in x_1d.columns if col.startswith('zscore_') or col.startswith('ratio_')]
    other_columns = [col for col in x_1d.columns if col not in zscore_ratio_columns]
    reordered_columns = other_columns + zscore_ratio_columns

    x_1d = x_1d[reordered_columns]
    x_2d = None
    y = deepcopy(x_1d['label'])

    #################################################
    #------------ Define search spaces -------------#
    #################################################
    #---   Decoding search spaces of feature selection ---#
    columns_to_remove = ['Patient', 'Stage','label']
    x_1d = x_1d.drop(columns=columns_to_remove)
    x_1d = remove_highly_correlated_features(x_1d)
    x_1d = remove_near_zero_variance_features(x_1d)
    x_1d_feature = x_1d.columns.tolist()
    x_2d_feature_0 = None
    x_2d_feature_1 = None
    
    #---   Decoding search spaces of NAS and HPO    ---#
    hyperparameter_spaces = [
        Categorical([2,4,6,8], name = 'trsf_hidden_each_head'), 
        Categorical([2,4], name = 'head'), 
        Integer(1,2, name = 'trsf_num_layer'),
        Integer(1,10, name = 'patience'), 
        Real(1e-1, 0.3, prior='log-uniform', name = 'trsf_dropout'), 
        Real(1e-1, 0.3, prior='log-uniform', name = 'fc_dropout'),
        Integer(1,5, name = 'num_fc_layers'),
        Real(1e-4, 1e-1, prior='log-uniform', name = 'learning_rate')
        ]

    determin_hyp={'num_output_nodes':2,'flag_1d': True, 'flag_2d': False, 'trsf_num_layer':1, 'num_fc_layers':1}

    #################################################
    #--------------- Other settings ----------------#
    #################################################
    max_len_2d, batch_size, num_outer_split, frac_inner_val, feature_batch, n_calls  = args.max_len_2d, \
        args.batch_size, args.num_outer_split, args.frac_inner_val, args.feature_batch, args.n_calls
    base_gp_path = script_directory+'/../'+args.optimized_result_path
    #---   Model defined in utils    ---#
    model_class = AutoTransformer
    
    #################################################
    #------------ Bayesian optimization ------------#
    #################################################
    
    performance = pd.DataFrame(columns=['repeats', 'set', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc'])

    for j in range(args.start_repeat, args.num_repeats):
        print('path:',path)
        seed = args.seed+j
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        KF = StratifiedKFold(5)
        for i, (train_val_index, test_index) in enumerate(KF.split(np.zeros(len(y)),y)):    
            repeat_i = 5*j+i
            print('repeat_i', repeat_i)
            gp_path = base_gp_path+str(repeat_i)+'.pkl'
            print('gp_path:',gp_path)
            x_1d_train, y_train = x_1d.iloc[train_val_index], y.iloc[train_val_index]
            x_1d_test, y_test = x_1d.iloc[test_index], y.iloc[test_index]

            gp, search_spaces, feature_search_spaces_len = auto_hpo_nnd_fs(hyperparameter_spaces, n_calls, model_class, determin_hyp, \
                x_1d_feature, x_2d_feature_0, x_2d_feature_1, x_1d_train, x_2d, y_train, \
                batch_size, feature_batch, num_outer_split, frac_inner_val, max_len_2d)
            params = {search_spaces[i].name: gp.x[i] for i in range(len(gp.x))}
            feature_params = {key: value for key, value in params.items() if key.startswith('feature')}
            f_1d, f_2d_0, f_2d_1 = int2feature(feature_params, feature_search_spaces_len, x_1d_feature, x_2d_feature_0, x_2d_feature_1)
            print('Selected feature: (1d){}, (2d_0){}, (2d_1){}'.format(f_1d, f_2d_0, f_2d_1))
            #---  Extract data with selected feature ---#
            x_1d_train_fs, x_2d_train_fs = fs_dataset(x_1d_train, None, f_1d, f_2d_0, f_2d_1)
            x_1d_test_fs, x_2d_test_fs = fs_dataset(x_1d_test, None, f_1d, f_2d_0, f_2d_1)

            train_index, val_index = next(StratifiedShuffleSplit(n_splits=1,test_size=1/9, random_state=seed).split(np.zeros(len(y_train)), y_train))
            ros = RandomOverSampler(sampling_strategy='auto', random_state=seed)
            train_index, _ = ros.fit_resample(train_index.reshape(-1, 1), y_train.iloc[train_index])
            train_index = train_index.flatten()
            dataset = FeatureSelectionDataset(x_1d_train_fs, x_2d_train_fs, y_train, max_len_2d)
            train_dataset = Subset(dataset, train_index)
            train_loader = DataLoader(train_dataset, batch_size=batch_size)
            val_dataset = Subset(dataset, val_index)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

            test_dataset = FeatureSelectionDataset(x_1d_test_fs, x_2d_test_fs, y_test, max_len_2d)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)

            _,tmp_2d,_ = next(iter(val_loader))

            #--- Construct model with optimized hyperparameters ---#
            torch.cuda.empty_cache()
            gc.collect()

            model_params = {key: value for key, value in params.items() if not key.startswith('feature')}
            model_params.update(determin_hyp)
            len_2d_0, len_2d_1 = len_2d_feature(val_loader)
            model_params.update({'len_1d':len(f_1d), 'len_2d':len_2d_0, 'dim_2d_input':len_2d_1})
            print('Model hyperparameters:', model_params)
            model = model_class(model_params)
            model.apply(init_weights)
            model.train_val(train_loader, val_loader)

            #--- Predict ---#
            y_hat_train, y_true_train, train_loss = model.test(train_loader)
            y_hat_val, y_true_val, val_loss = model.test(val_loader)
            y_hat_test, y_true_test, test_loss = model.test(test_loader)

            #--- Evaluate ---#print('-'*40)
            print('Training performance')
            evaluate(y_hat_train, y_true_train, average = 'binary')
            print('Validation performance')
            accuracy, precision, recall, f1, roc_auc, pr_auc = evaluate(y_hat_val, y_true_val, average = 'binary', flag_soft=True)
            val = pd.DataFrame([[repeat_i, 'validation', accuracy, precision, recall, f1, roc_auc, pr_auc]], columns=['repeats', 'set', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc'])
            print('Test performance')
            accuracy, precision, recall, f1, roc_auc, pr_auc = evaluate( y_hat_test, y_true_test, average = 'binary', flag_soft=True)
            test = pd.DataFrame([[repeat_i, 'test', accuracy, precision, recall, f1, roc_auc, pr_auc]], columns=['repeats', 'set', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc'])
            performance = pd.concat([performance,val,test], ignore_index=True, axis=0)

            performance.to_csv(path, index=False)
        
    val_performance = performance[performance.set=='validation']
    print('Overall performance on validation set:')
    print('accuracy: {:6.4f}, precision: {:6.4f}, recall {:6.4f} f1: {:6.4f}, roc_auc: {:6.4f}, pr_auc: {:6.4f}'.format(val_performance.accuracy.mean(), 
    val_performance.precision.mean(), val_performance.recall.mean(), val_performance.f1.mean(), val_performance.roc_auc.mean(), val_performance.pr_auc.mean()))

    test_performance = performance[performance.set=='test']
    print('Overall performance on test set:')
    print('accuracy: {:6.4f}, precision: {:6.4f}, recall {:6.4f} f1: {:6.4f}, roc_auc: {:6.4f}, pr_auc: {:6.4f}'.format(test_performance.accuracy.mean(), 
    test_performance.precision.mean(), test_performance.recall.mean(), test_performance.f1.mean(), test_performance.roc_auc.mean(), test_performance.pr_auc.mean()))

        
if __name__ == '__main__':
    main()