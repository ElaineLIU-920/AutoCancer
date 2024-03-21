from utils import *
from copy import deepcopy
from sklearn.model_selection import StratifiedShuffleSplit

import random
import skopt
import argparse
import os
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=920,
                        help='random seed (default: 920)')
    parser.add_argument('--max_len_2d', type=int, default=0,
                        help='max length of SNVs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size (default: 256)')
    parser.add_argument('--num_outer_split', type=int, default=5,
                        help='parameter for data splitting')
    parser.add_argument('--frac_inner_val', type=float, default=0.25,
                        help='parameter for data splitting')
    parser.add_argument('--feature_batch', type=int, default=10,
                        help='the maximum value of a feature group for FS (default: 10)')
    parser.add_argument('--n_calls', type=int, default=50,
                        help='the maximum evolution step for Bayesian optimization (default: 100)')
    parser.add_argument('--num_repeats', type=int, default=10,
                        help='the number of repeats')
    args = parser.parse_args()

    #################################################
    #---------------- Seed setup ----------------#
    #################################################
    seed = args.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #################################################
    #------------------ Load Data ------------------#
    #################################################
    script_directory = os.path.dirname(os.path.abspath(__file__))
    import datetime
    import time
    today = datetime.date.today()
    t = time.localtime()
    path = script_directory+'/../result/comparison_result/pancancer/AutoCancer.csv'
    x_path = script_directory+'/dataset/pancancer_dataset.csv'

    x_1d = pd.read_csv(x_path,index_col=0)
    x_2d = None
    y = deepcopy(x_1d['label'])

    #################################################
    #------------ Define search spaces -------------#
    #################################################

    #---   Decoding search spaces of feature selection ---#
    columns_to_remove = ['cancer_type_label','label']
    x_1d_feature = x_1d.drop(columns=columns_to_remove).columns.tolist()
    x_2d_feature_0 = None
    x_2d_feature_1 = None # 'snv_discription'
    
    #---   Decoding search spaces of NAS and HPO    ---#
    hyperparameter_spaces = [
        Real(1e-1, 0.3, prior='log-uniform', name = 'initial_dropout'),
        Integer(1,5, name = 'num_fc_layers'),
        Integer(1,10, name = 'patience'), 
        Categorical([nn.Sigmoid(), nn.ReLU(), nn.LeakyReLU(), nn.LeakyReLU()], name='act'),
        Real(1e-4, 1e-1, prior='log-uniform', name = 'learning_rate')
        ]
    determin_hyp={'num_output_nodes':2,'flag_1d': True, 'flag_2d': False}
    
    #################################################
    #--------------- Other settings ----------------#
    #################################################
    max_len_2d, batch_size, num_outer_split, frac_inner_val, feature_batch, n_calls  = args.max_len_2d, \
        args.batch_size, args.num_outer_split, args.frac_inner_val, args.feature_batch, args.n_calls
    #---   Model defined in utils    ---#
    model_class = AutoMLP
    
    #################################################
    #------------ Bayesian optimization ------------#
    #################################################
    KF = StratifiedKFold(10,shuffle=True, random_state=920)
    performance = pd.DataFrame(columns=['repeats', 'set', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc'])
    for i, (train_val_index, test_index) in enumerate(KF.split(np.zeros(len(y)),y)):
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


        train_index, val_index = next(StratifiedShuffleSplit(n_splits=1,test_size=1/5, random_state=seed).split(np.zeros(len(y_train)), y_train))
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
        val = pd.DataFrame([[i, 'validation', accuracy, precision, recall, f1, roc_auc, pr_auc]], columns=['repeats', 'set', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc'])
        print('Test performance')
        accuracy, precision, recall, f1, roc_auc, pr_auc = evaluate( y_hat_test, y_true_test, average = 'binary', flag_soft=True)
        test = pd.DataFrame([[i, 'test', accuracy, precision, recall, f1, roc_auc, pr_auc]], columns=['repeats', 'set', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc'])
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