import fedml
import os
# contest
from .trade.data_loader_custom import load_partition_data_trade

def load(args):
    return load_synthetic_data(args)

def load_synthetic_data(args):
    dataset_name = args.dataset

    if dataset_name == "trade":
        (
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ) = load_partition_data_trade(
            args,
            args.batch_size
        )
    elif dataset_name == "custom":
        (
            train_data_num,  # 全部客户端共有多少个训练数据
            test_data_num,
            train_data_global, # 全部客户端的数据
            test_data_global,
            train_data_local_num_dict, # 用户。记录每个client下多少个训练数据
            train_data_local_dict, # 用户
            test_data_local_dict,
            class_num,  # 用户。y的标签个数
        ) = load_partition_data_trade(  # client_id设置为rank-1，从0开始
            args,
            args.batch_size
        )

    else:
        a = 2

    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ]

    return dataset, class_num
