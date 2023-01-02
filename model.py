import torch
import numpy as np
import torch.nn.functional as F
from torch import nn

import warnings
warnings.filterwarnings("ignore")

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask=None):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores = torch.exp(scores)
        if attn_mask is not None:
            scores = scores * attn_mask
        attn = scores / (torch.sum(scores, dim=-1, keepdim=True) + 1e-8)

        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model  # 300
        self.n_heads = n_heads  # 20
        self.d_k = d_k  # 20
        self.d_v = d_v  # 20

        self.W_Q = nn.Linear(d_model, d_k * n_heads)  # 300, 400
        self.W_K = nn.Linear(d_model, d_k * n_heads)  # 300, 400
        self.W_V = nn.Linear(d_model, d_v * n_heads)  # 300, 400

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, Q, attn_mask=None):
        batch_size, seq_len, hidden_size = Q.size()

        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v_s = self.W_V(Q).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        context, attn = ScaledDotProductAttention(self.d_k)(q_s, k_s, v_s, attn_mask)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.n_heads * self.d_v)
        )
        return context

class AttentionPooling(nn.Module):
    def __init__(self, d_h, hidden_size, drop_rate):
        super(AttentionPooling, self).__init__()
        self.att_fc1 = nn.Linear(d_h, hidden_size // 2)
        self.att_fc2 = nn.Linear(hidden_size // 2, 1)
        self.drop_layer = nn.Dropout(p=drop_rate)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.zeros_(m.bias)

    def forward(self, x, attn_mask=None):

        bz = x.shape[0]
        e = self.att_fc1(x)  # (bz, seq_len, 200)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e)  # (bz, seq_len, 1)

        alpha = torch.exp(alpha)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)

        x = torch.bmm(x.permute(0, 2, 1), alpha)
        x = torch.reshape(x, (bz, -1))  # (bz, 400)
        return x

class User_Model(nn.Module):
    def __init__(self,hidden_size,bin_num):
        super(User_Model, self).__init__()
        self.hidden_size = hidden_size
        self.bin_num = bin_num
        self.build_model()

    def build_model(self):
        self.embeddings = nn.ModuleList([nn.Embedding(10,self.hidden_size),
                                        nn.Embedding(15,self.hidden_size),
                                        nn.Embedding(330,self.hidden_size),
                                        nn.Embedding(15,self.hidden_size),
                                        ])
        self.trans = nn.Sequential(
            nn.Linear(4*self.hidden_size+1,self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size,self.hidden_size),
        )
        self.position_embedding = nn.Embedding(50,self.hidden_size)
        #self.user_attention = AttentionPooling(self.hidden_size,self.hidden_size,0.5)
        self.shop_attention = AttentionPooling(self.hidden_size,self.hidden_size,0.5)
        self.dropout = nn.Dropout(0.2)

    def forward(self,feats):
        #feats : [batch,user_num,4]
        batch_size, user_num, f_num = feats.shape
        feats = torch.tensor(feats)

        feats = feats.view(-1,f_num)
        feats_embedding = []
        for i in range(4):
            feats_embedding.append(self.dropout(self.embeddings[i](feats[:,i].long())))
        feats_embedding = torch.cat([torch.cat(feats_embedding,1),feats[:,4:]],1)
        #[batch*user_num,dim]
        feats_embedding = self.trans(feats_embedding)
        feats_embedding = feats_embedding.view(batch_size,user_num,-1)
        feats_embedding = feats_embedding + self.dropout(self.position_embedding(torch.tensor([i for i in range(user_num)], dtype=torch.int64).expand(batch_size,user_num)))
        feats_embedding = self.shop_attention(feats_embedding)

        return feats_embedding




class Net(nn.Module):
    def __init__(self, hidden_size=16, num_layers=2, lr=0.001):
        super(Net, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bin_num=10
        self.val_score=torch.nn.Parameter(torch.tensor(0,dtype=torch.float32))
        self.build_model()

    def build_model(self):
        self.user_model = User_Model(self.hidden_size,self.bin_num)
        self.dropout = nn.Dropout(0.2)
        input_size = 4+self.bin_num+self.hidden_size
        self.model = nn.Sequential(
            nn.Linear(input_size,input_size//2),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(input_size//2,self.hidden_size),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(self.hidden_size,1)
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, feats):
        embeds = []
        embeds.append(torch.cat([torch.tensor(feats[0], dtype=torch.float32),torch.tensor(feats[1], dtype=torch.float32)],1))
        user_embeds = self.user_model(feats[-1])
        embeds.append(user_embeds)
        embeds = torch.cat(embeds, 1)

        logits = self.model(embeds)
        preds = self.sigmoid(logits)
        return logits, preds
