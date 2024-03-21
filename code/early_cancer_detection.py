
import random
import skopt
import numpy as np
import pandas as pd
from copy import deepcopy

from sklearn.model_selection import StratifiedShuffleSplit

import argparse
from utils import *

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1107,
                        help='random seed (default: 1105)')
    parser.add_argument('--max_len_2d', type=int, default=84,
                        help='max length of SNVs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size (default: 256)')
    parser.add_argument('--test_ratio', type=int, default=0.25,
                        help='parameter for data splitting')
    parser.add_argument('--feature_batch', type=int, default=10,
                        help='the maximum value of a feature group for FS (default: 10)')
    parser.add_argument('--x_1d_train_path', default='./dataset/NSCLC_cohort/x_1d_train.pkl',
                        help='path for loading the 1D training data including label')
    parser.add_argument('--x_2d_train_path', default='./dataset/NSCLC_cohort/x_2d_train.pkl',
                        help='path for loading the 2D training data')
    parser.add_argument('--x_1d_test_path', default='./dataset/NSCLC_cohort/x_1d_test.pkl',
                        help='path for loading the 1D test data including label')
    parser.add_argument('--x_2d_test_path', default='./dataset/NSCLC_cohort/x_2d_test.pkl',
                        help='path for loading the 2D test data')
    parser.add_argument('--optimized_result_path', default='./result/optimized_result.pkl',
                        help='path for saving the optimized result')
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
    #--------------- Other settings ----------------#
    #################################################
    max_len_2d, batch_size, test_ratio, feature_batch = args.max_len_2d, args.batch_size, args.test_ratio, args.feature_batch

    #---   Model defined in utils    ---#
    model_class = AutoTransformer

    #################################################
    #------------------ Load Data ------------------#
    #################################################
    script_directory = os.path.dirname(os.path.abspath(__file__))
    x_1d_train = pd.read_pickle(script_directory+'/../'+args.x_1d_train_path)
    x_2d_train = pd.read_pickle(script_directory+'/../'+args.x_2d_train_path)
    y_train = deepcopy(x_1d_train['label'])

    x_1d_test = pd.read_pickle(script_directory+'/../'+args.x_1d_test_path)
    x_2d_test = pd.read_pickle(script_directory+'/../'+args.x_2d_test_path)
    y_test = deepcopy(x_1d_test['label'])

    #################################################
    #------------ Define search spaces -------------#
    #################################################
    #---   Decoding search spaces of feature selection ---#
    columns_to_remove = ['split_info', 'Histology label', 'label', 'stage_label']
    x_1d_feature = x_1d_train.drop(columns=columns_to_remove).columns.tolist()
    x_2d_feature_0 = None # for no feature selection on this dimension
    x_2d_feature_1 = x_2d_train.columns.tolist() # 'snv_discription'
    
    #---   Decoding search spaces of NAS and HPO    ---#
    hyperparameter_spaces = [
        Categorical([16,32,64], name = 'trsf_hidden_each_head'), 
        Categorical([2,4,6,8,10], name = 'head'), 
        Integer(1,5, name = 'trsf_num_layer'),
        Integer(1,10, name = 'patience'), 
        Real(1e-1, 0.3, prior='log-uniform', name = 'trsf_dropout'), 
        Real(1e-1, 0.3, prior='log-uniform', name = 'fc_dropout'),
        Integer(1,5, name = 'num_fc_layers'),
        Real(1e-4, 1e-1, prior='log-uniform', name = 'learning_rate')
        ]

    determin_hyp={'num_output_nodes':2,'flag_1d': True, 'flag_2d': True}

    feature_search_spaces, feature_search_spaces_len = construct_feature_search_space(x_1d_feature, x_2d_feature_0, x_2d_feature_1, feature_batch)
    search_spaces = feature_search_spaces + hyperparameter_spaces
    
    #################################################
    #------------ Load optimized result ------------#
    #################################################
    #--- Extract optimized result ---#
    optimized_path = script_directory+'/../'+args.optimized_result_path
    gp = skopt.load(optimized_path)
    params = {search_spaces[i].name: gp.x[i] for i in range(len(gp.x))}

    #--- Extract selected feature ---#    
    feature_params = {key: value for key, value in params.items() if key.startswith('feature')}
    f_1d, f_2d_0, f_2d_1 = int2feature(feature_params, feature_search_spaces_len, x_1d_feature, x_2d_feature_0, x_2d_feature_1)
    print('-'*40)
    print('Selected feature:\n(1d){},\n(2d){}'.format(f_1d,f_2d_1))
    #---  Extract data with selected feature ---#
    x_1d_train_fs, x_2d_train_fs = fs_dataset(x_1d_train, x_2d_train, f_1d, f_2d_0, f_2d_1)
    x_1d_test_fs, x_2d_test_fs = fs_dataset(x_1d_test, x_2d_test, f_1d, f_2d_0, f_2d_1)

    #################################################
    #------------------ Diagnose -------------------#
    #################################################
    #--- Split training and validation dataset ---#
    train_index, val_index = next(StratifiedShuffleSplit(n_splits=1,test_size=test_ratio, random_state=seed).split(np.zeros(len(y_train)), y_train))
    ros = RandomOverSampler(sampling_strategy='auto', random_state=seed)
    train_index, _ = ros.fit_resample(train_index.reshape(-1, 1), y_train.iloc[train_index])
    train_index = train_index.flatten()
    dataset = FeatureSelectionDataset(x_1d_train_fs, x_2d_train_fs, y_train, max_len_2d)
    train_dataset = Subset(dataset, train_index)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataset = Subset(dataset, val_index)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    _,tmp_2d,_ = next(iter(val_loader))
    #--- Construct test set with independant dataset ---#
    test_dataset = FeatureSelectionDataset(x_1d_test_fs, x_2d_test_fs, y_test, max_len_2d)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    #--- Construct model with optimized hyperparameters ---#
    model_params = {key: value for key, value in params.items() if not key.startswith('feature')}
    model_params.update(determin_hyp)
    model_params.update({'len_1d':len(f_1d), 'len_2d':tmp_2d.shape[1], 'dim_2d_input':tmp_2d.shape[2]})
    print('-'*40)
    print('Model hyperparameters:', model_params)
    model = model_class(model_params)
    model.apply(init_weights)
    model.train_val(train_loader, val_loader)

    #--- Predict ---#
    y_hat_train, y_true_train, train_loss = model.test(train_loader)
    y_hat_val, y_true_val, val_loss = model.test(val_loader)
    y_hat_test, y_true_test, test_loss = model.test(test_loader)

    #--- Evaluate ---
    print('Training performance')
    evaluate(y_hat_train, y_true_train, average = 'binary')
    print('Validation performance')
    accuracy, precision, recall, f1, roc_auc, pr_auc = evaluate(y_hat_val, y_true_val, average = 'binary')
    # val = pd.DataFrame([[i, 'validation', accuracy, precision, recall, f1, roc_auc, pr_auc]], columns=['repeats', 'set', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc'])
    print('Test performance')
    accuracy, precision, recall, f1, roc_auc, pr_auc = evaluate( y_hat_test, y_true_test, average = 'binary')
    # test = pd.DataFrame([[i, 'test', accuracy, precision, recall, f1, roc_auc, pr_auc]], columns=['repeats', 'set', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc'])

if __name__ == '__main__':
    main()