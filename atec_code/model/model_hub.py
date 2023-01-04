import fedml
import os
import torch
import importlib
# from .mlp import Net

def create(args, output_dim):
    model_name = args.model
    model_dir = "model." + model_name
    model_file = importlib.import_module(model_dir) # 写成可导入的形式
    if args.dataset == 'trade' or args.dataset=='custom':  # 比赛使用的模型：用户若是给网络传递参数，则直接在model定义时，写死就好了
       model = model_file.Net()
    print(model)
    return  model




