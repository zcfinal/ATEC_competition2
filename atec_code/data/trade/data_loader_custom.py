import json
import os
import numpy as np
import wget
import zipfile
import logging

from torchvision import datasets, transforms
import torch

import sys
import os
import pickle
import sklearn
import numpy as np
from collections import Counter
from sklearn.preprocessing import KBinsDiscretizer

from .client_partition_data.client_partition_data import gen_train_data
#s

def load_partition_data_trade(
    args, batch_size
):

    train_data_local_num_dict = dict()
    train_data_local_dict = dict()
    if args.rank == 0: # 
        train_data_local_num_dict[args.client_num_in_total] = 10
        train_data_local_dict[args.client_num_in_total] = [ [0] ]
        train_data_global = None
        
    else: 
        for i in os.listdir(args.data_cache_dir):
            if "buyer_feature.csv" in i:
                c_path_buyer_features = os.path.join(args.data_cache_dir, "trade_buyer_feature.csv")
            if "edge_feature.csv" in i:
                c_path_edge_features = os.path.join(args.data_cache_dir, "trade_edge_feature.csv")
            if "train_label.csv" in i:
                c_path_train_label = os.path.join(args.data_cache_dir, "train_label.csv")
                
        # 
        train_batch, train_num = gen_train_data(c_path_buyer_features, c_path_edge_features, c_path_train_label)
    
        train_data_local_num_dict[args.rank-1] = train_num #
        train_data_local_dict[args.rank-1] = train_batch # 
        train_data_global = train_batch
    
    train_data_num = 30503 
    if args.rank == 0: 
        '''
        '''
        test_data_num = 100  # client没有测试集，则随便赋值
        test_data_local_dict = dict()
        test_data_local_dict[args.rank-1] = [ [0]  ]
        test_data_global = None
    else: # 若不是server.
        test_data_num = 100  # client没有测试集，则随便赋值
        test_data_local_dict = dict()
        test_data_local_dict[args.rank-1] = [ [0]  ]
        test_data_global = None

    class_num = 2
   
    return (
        train_data_num,         #
        test_data_num,          #
        train_data_global,      #
        test_data_global,       #
        train_data_local_num_dict,
        train_data_local_dict,   #
        test_data_local_dict,    #
        class_num,               #
    )
