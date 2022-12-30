import wandb
import copy

import warnings
warnings.filterwarnings("ignore")

class My_ClassificationAggregator():
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