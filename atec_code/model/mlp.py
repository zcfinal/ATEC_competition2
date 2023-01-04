import torch
import numpy as np
import torch.nn.functional as F
from torch import nn


class Net(nn.Module):
	def __init__(self, hidden_size=8, num_layers=2, lr=0.001):
		super(Net, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.build_model()

	def build_model(self):
		self.embeds_feat_trade_num = nn.Embedding(11, self.hidden_size)
		self.embeds_feat_avg_trade_amount = nn.Embedding(11, self.hidden_size)
		self.embeds_feat_return_num = nn.Embedding(11, self.hidden_size)

		input_size = self.hidden_size*3 + 10
		self.fc1 = torch.nn.Linear(input_size, 16)
		self.relu = torch.nn.ReLU()
		self.fc2 = torch.nn.Linear(16, 1)
		self.sigmoid = torch.nn.Sigmoid()

	def forward(self, feats):
		embeds = []
		embeds.append(self.embeds_feat_trade_num(torch.tensor(feats[1], dtype=torch.int64)))
		embeds.append(self.embeds_feat_avg_trade_amount(torch.tensor(feats[2], dtype=torch.int64)))
		embeds.append(self.embeds_feat_return_num(torch.tensor(feats[3], dtype=torch.int64)))
		embeds.append(torch.tensor(feats[0], dtype=torch.float32))
		embeds = torch.cat(embeds, 1)

		logits = self.fc2(self.relu(self.fc1(embeds)))
		preds = self.sigmoid(logits)
		return logits, preds