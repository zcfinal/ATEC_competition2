import torch
import os
from torch import nn
import logging
import numpy as np
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")
import copy
from sklearn.metrics import roc_auc_score
def calc_auc(preds, labels): 
	return roc_auc_score(labels, preds)

class Minibatch:
	def __init__(self, train_feats, train_labels, batch_size, shuffle=True):
		self.train_feats = train_feats
		self.train_labels = train_labels
		self.batch_size = batch_size
		self.total_cnt = len(self.train_feats[0])
		self.feat_cnt = len(self.train_feats)
		self.indices = np.arange(self.total_cnt)
		if shuffle:
			self.indices = np.random.permutation(self.indices)
		self.cur_idx = 0

	def next_batch(self):
		cur_indices = self.indices[self.cur_idx:self.cur_idx+self.batch_size]
		batch_feats = []
		for i in range(self.feat_cnt):
			batch_feats.append(self.train_feats[i][cur_indices])
		batch_labels = self.train_labels[cur_indices]
		self.cur_idx += self.batch_size
		return batch_feats, batch_labels

	def has_next(self):
		return self.cur_idx < self.total_cnt




class My_ClassificationTrainer():
	def get_model_params(self):
		return self.model.cpu().state_dict()

	def set_model_params(self, model_parameters):
		self.model.load_state_dict(model_parameters)
	
	def train(self, train_data, device, args):
		model = self.model
		model.to(device)
		model.train()
		optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)
		auc_best = 0
		total_labels = train_data[1]
		total_feats = train_data[0]
		split_idx = int(len(total_labels)*0.8)
		train_feats = []
		valid_feats = []

		for feats in total_feats:
			train_feats.append(feats[:split_idx])
			valid_feats.append(feats[split_idx:])
		train_labels = total_labels[:split_idx]
		valid_labels = total_labels[split_idx:]
		#data_train = train_data

		for epoch in range(2):
			minibatch = Minibatch(train_feats, train_labels, 64, shuffle=True)
			train_losses = []
			#p=[]
			#l=[]
			#train_data=data_train
			#train_data.cur_idx=0
			model.train()
			while minibatch.has_next():
				feats, label = minibatch.next_batch()
				#l.extend(label)
				model.zero_grad()
				logits, preds = model(feats)
				#p.extend(preds.detach().numpy().reshape([-1]))
				criterion = nn.BCEWithLogitsLoss(reduction='none')

				loss = criterion(logits.view(-1), torch.tensor(label, dtype=torch.float32))

				mask = 1.2*(torch.tensor(label, dtype=torch.float32)==1).float() +(torch.tensor(label, dtype=torch.float32)==0).float()
				loss = torch.mean(mask*loss)
				loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
				optimizer.step()
				train_losses.append(loss.detach().numpy())
			model.eval()
			with torch.no_grad():
				l, valid_preds = model(valid_feats)
			auc = calc_auc(valid_preds.numpy().reshape([-1]), valid_labels)
			#auc = calc_auc(np.array(p).reshape([-1]), np.array(l))
			if auc > auc_best:
				auc_best = auc
				print('Saving model ...')
				#torch.save(model.state_dict(), '%s' % args.global_model_file_path)
			model.val_score = torch.nn.Parameter(torch.tensor(auc,dtype=torch.float32))
			print('[Epoch {}] TRAIN: loss = {:.4f}'.format(epoch+1, np.mean(train_losses)))
			print('[Epoch {}] VALIDATION: auc = {:.4f}'.format(epoch+1, auc))
			print('[Epoch {}] best: auc = {:.4f}'.format(epoch+1, auc_best))