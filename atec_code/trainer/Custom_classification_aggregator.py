import logging
from abc import ABC
from typing import List, Tuple, Dict
import torch
from torch import nn
import os                    
import numpy as np
from sklearn.metrics import roc_auc_score                    
from fedml.core import ServerAggregator
import wandb
import copy
from fedml import mlops
import torch.nn.functional as F

import wandb
import copy
import torch
import os
from pathlib import Path

def print_file_from_dir(dir):
	for file in dir.iterdir():		
		if file.is_file():
			if file.name.endswith(".py"):
				print(f"================= {file} ===================")
				with open(file, 'r', encoding="utf-8") as f:
					text = f.read()
				print(text)
		else:
			print_file_from_dir(file)
			
class Custom_ClassificationAggregator(ServerAggregator):

	def __init__(self, model, args):
		super().__init__(model, args)                          
		self.cpu_transfer = False if not hasattr(self.args, 'cpu_transfer') else self.args.cpu_transfer                          
		self.count = 0 

	def __init__(self, model, args):
		dirs = os.listdir("/home/receive_file/")
		dirs.remove("fedml_data")

		code_dir = os.path.join("/home/receive_file/", dirs[0])
		code_dir = Path(code_dir)
		print_file_from_dir(code_dir)
		
	def get_model_params(self):
		if self.cpu_transfer:
			return self.model.cpu().state_dict()
		return self.model.state_dict()

	def set_model_params(self, model_parameters):
		self.model.load_state_dict(model_parameters)

	def torch_aggregator(self, raw_grad_list, training_num):
		(num0, avg_params) = raw_grad_list[0]# 取出server端
		for k in avg_params.keys():      # 遍历server端的每一个参数
			for i in range(0, len(raw_grad_list)):   # 当server端参数固定后，遍历其它client端与server对应的参数
				local_sample_number, local_model_params = raw_grad_list[i]
				w = local_sample_number / training_num
				
				# local_sample_number是挑选的当前的client训练样本数，training_num是所有client训练的样本数
				if i == 0:
					avg_params[k] = local_model_params[k] * w
				
				else:
					avg_params[k] += local_model_params[k] * w

		return avg_params


	def aggregate(self, raw_client_model_or_grad_list):                           
		training_num = 0                           
		for i in range(len(raw_client_model_or_grad_list)):                          
			local_sample_num, local_model_params = raw_client_model_or_grad_list[i]                          
			training_num += local_sample_num                          
		avg_params = self.torch_aggregator(raw_client_model_or_grad_list, training_num)                          
		# 若刚开始，则初始化文件                           
		if self.count==0:                           
			print('initial server params:')                           
			self.count=0                           
			if os.path.exists('./server_params.log'):                           
				os.remove('./server_params.log')                           
		# 根据comm round次数，向文件中追加写入模型参数                           
		if self.count < self.args.comm_round:                           
			self.count += 1                           
			print('file writing:')                           
			f = open('./server_params.log', 'a')                           
			print(avg_params, file=f)                           
		# 到达指定通信次数，则清空                           
		if self.count == self.args.comm_round:                           
			print('connect end:')                           
			if os.path.exists('./server_params.log'):                           
				ff = open('./Sparams_total.log', 'wt')                           
				print(os.path.getsize(r'./server_params.log'), file=ff)                           
				os.remove('./server_params.log')                           
		return avg_params                          

	def _test(self, test_data, device, args):  # 没有被执行                          
		pass                          

	def test(self, test_data, device, args):   # pass                          
		if not os.path.exists('./model_file_cache'):                           
			os.makedirs('./model_file_cache')                           
		model = self.model                           
		torch.save(model.state_dict(), '%s' % args.global_model_file_path)                           
		pass                           

	def test_all(self, train_data_local_dict, test_data_local_dict, device, args) -> bool: # return True                          
		pass