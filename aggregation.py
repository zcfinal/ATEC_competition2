import wandb
import copy
import torch

class My_ClassificationAggregator():
	def get_model_params(self):
		if self.cpu_transfer:
			return self.model.cpu().state_dict()
		return self.model.state_dict()

	def set_model_params(self, model_parameters):
		self.model.load_state_dict(model_parameters)

	def torch_aggregator(self, raw_grad_list, training_num):
		if not hasattr(self,'optimizer'):
			print('create optimizer')
			self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.05, weight_decay=0.001)
		if hasattr(self,'aggregation_num'):
			self.aggregation_num+=1
		else:
			self.aggregation_num=1
		print(f'round:{self.aggregation_num}')

		tmp_param = copy.deepcopy(self.model.state_dict())
		self.optimizer.zero_grad()

		val_score = 0

		for k,param in self.model.named_parameters():
			for i in range(0, len(raw_grad_list)):
				local_sample_number, local_model_params = raw_grad_list[i]
				w = local_sample_number / training_num
				if k=='val_score':
					val_score += local_model_params[k]*w
				if param.grad is None:
					param.grad = (tmp_param[k]-local_model_params[k]) * w
				else:
					param.grad += (tmp_param[k]-local_model_params[k]) * w
		
		self.optimizer.step()
		avg_params = self.model.state_dict()
		avg_params['val_score'] = val_score

		if hasattr(self,'val_score') and avg_params['val_score']>self.val_score:
			self.val_score=avg_params['val_score']
			self.best_model = avg_params
		elif not hasattr(self,'val_score'):
			self.val_score=avg_params['val_score']
			self.best_model = avg_params

		print(f'best val score:{self.val_score}')
		if self.aggregation_num==200:
			print('load best')
			self.set_model_params(self.best_model)
			return self.best_model

		return avg_params