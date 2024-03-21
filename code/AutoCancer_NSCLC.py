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
    parser.add_argument('--seed', type=int, default=1105,
                        help='random seed (default: 1105)')
    parser.add_argument('--max_len_2d', type=int, default=84,
                        help='max length of SNVs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size (default: 256)')
    parser.add_argument('--num_outer_split', type=int, default=5,
                        help='parameter for data splitting')
    parser.add_argument('--frac_inner_val', type=int, default=0.25,
                        help='parameter for data splitting')
    parser.add_argument('--feature_batch', type=int, default=10,
                        help='the maximum value of a feature group for FS (default: 10)')
    parser.add_argument('--n_calls', type=int, default=100,
                        help='the maximum evolution step for Bayesian optimization (default: 100)')
    parser.add_argument('--x_1d_train_path', default='./dataset/x_1d_train.pkl',
                        help='path for loading the 1D training data including label')
    parser.add_argument('--x_2d_train_path', default='./dataset/x_2d_train.pkl',
                        help='path for loading the 2D training data')
    parser.add_argument('--optimized_result_path', default='./result/optimized_result-new.pkl',
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
    #------------------ Load Data ------------------#
    #################################################
    script_directory = os.path.dirname(os.path.abspath(__file__))
    x_1d = pd.read_pickle(script_directory+'/../'+args.x_1d_train_path)
    x_2d = pd.read_pickle(script_directory+'/../'+args.x_2d_train_path)
    y = deepcopy(x_1d['label'])

    #################################################
    #------------ Define search spaces -------------#
    #################################################

    #---   Decoding search spaces of feature selection ---#
    columns_to_remove = ['split_info', 'Histology label', 'label', 'stage_label']
    x_1d_feature = x_1d.drop(columns=columns_to_remove).columns.tolist()
    x_2d_feature_0 = None
    x_2d_feature_1 = x_2d.columns.tolist() # 'snv_discription'
    
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

    #################################################
    #--------------- Other settings ----------------#
    #################################################
    max_len_2d, batch_size, num_outer_split, frac_inner_val, feature_batch, n_calls  = args.max_len_2d, \
        args.batch_size, args.num_outer_split, args.frac_inner_val, args.feature_batch, args.n_calls
    gp_path = script_directory+'/../'+args.optimized_result_path
    #---   Model defined in utils    ---#
    model_class = AutoTransformer
    
    #################################################
    #------------ Bayesian optimization ------------#
    #################################################
    gp, search_spaces, feature_search_spaces_len = auto_hpo_nas_fs(hyperparameter_spaces, n_calls, model_class, determin_hyp, x_1d_feature, x_2d_feature_0, x_2d_feature_1, x_1d, x_2d, y, \
    batch_size, feature_batch, num_outer_split, frac_inner_val, max_len_2d)
    #---   Save optimized results    ---#
    for i in ['models', 'space', 'random_state', 'specs']:
        del gp[i]  
    skopt.dump(gp, gp_path)
    
if __name__ == '__main__':
    main()