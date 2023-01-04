import torch
import sys

import fedml
from fedml import FedMLRunner
from data.data_loader import load
from model.model_hub import create
from trainer.Custom_ClassificationTrainer import Custom_ClassificationTrainer
from trainer.Custom_classification_aggregator import Custom_ClassificationAggregator


if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, class_num = load(args)   # class_num=10,
    # create model and trainer
    model = create(args, output_dim=class_num)

    # 若用户自定义client trainer方式，则调用Custom_ClassificationTrainer.py
    if args.custom_client_trainer == True: # ”用户全部自定义client_trianer“
        print("client trainer ...`")
        trainer = Custom_ClassificationTrainer(model=model, args=args)    # client
    elif args.custom_client_trainer == False: # “列表选择形式” 与 ”用户自定义loss或optimizer“
        pass
    else:
        a=3

    # 若用户自定义server agg方式，则调用Custom_ClassificationAggregator.py
    if args.custom_server_agg == True:  # "用户全部自定义server的set/get param、agg"
        aggregator = Custom_ClassificationAggregator(model=model, args=args)  # server
    elif args.custom_server_agg == False: # ”列表选择server agg“ 与 "用户自定义agg"
        pass
    else:
        a=3

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model, trainer, aggregator)
    fedml_runner.run()
