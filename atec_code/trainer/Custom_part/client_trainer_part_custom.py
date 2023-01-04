import os                  
import torch                  
from torch import nn                  
import logging                  
def custom_train(self, train_data, device, args):                  
	model = self.model                  
	model.to(device)                  
	model.train()                  
	epoch_loss = []                  
	min_loss = 100000                  
	# 存储训练好的模型的文件                  
	if not os.path.exists('./model_file_cache'):                  
		os.makedirs('./model_file_cache') 
	criterion = nn.NLLLoss()
	optimizer = torch.optim.Adam(                 
		model.parameters(),                 
		lr=self.args.learning_rate,                 
		weight_decay=self.args.weight_decay,                 
		amsgrad=True,            )
	for epoch in range(1, args.epochs + 1):                   
		batch_loss = []                   
		for batch_idx, (data, target) in enumerate(train_data):                   
			data, target = data.to(device), target.to(device)                   
			model.zero_grad()                   
			output = model(data)                   
			loss = criterion(output, target)                   
			loss.backward()                   
			optimizer.step()                   
			if batch_idx % 10 == 0:                   
				print('Train Epoch: {} [{}/{} ({:.0f}%)]	Loss: {:.6f}'.format(                   
					epoch, batch_idx * len(data), len(train_data.dataset),                   
					100. * batch_idx / len(train_data), loss.item()))                   
			batch_loss.append(loss.item())                   
		# 当一个epoch执行结束后                   
		if len(batch_loss) == 0:                   
			epoch_loss.append(0.0)                   
		else:                   
			epoch_loss.append(sum(batch_loss) / len(batch_loss))                   
		# 输出截止到目前为止，loss的总和除以epoch的个数                   
		logging.info(                   
			'Client Index = {}	Epoch: {}	Loss: {:.6f}'.format(                   
			self.id, epoch, sum(epoch_loss) / len(epoch_loss)                   
			)                   
		)                   
		# 保存模型---找出训练集中最小的模型参数并保存                   
		if sum(batch_loss) < min_loss:                   
			min_loss = sum(batch_loss)                   
			torch.save(model, '%s' % args.global_model_file_path)    