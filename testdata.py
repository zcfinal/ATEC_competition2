import json
import os
import numpy as np
import torch
from collections import Counter
from sklearn.preprocessing import KBinsDiscretizer

import warnings
warnings.filterwarnings("ignore")
def train_feat_process(path_buyer_features, path_edge_features, path_train_label): # 开放给用户
	
	"""
	process trade buyer features
	"""
	bin_num=10
	user_column = {'buyer_id':0,
					'gender':1,
					'age':2,
					'city':3,
					'occupation':4}
	buyer_feat_set = {i:set() for i in range(5)}
	buyer_feats = {}
	is_first_line = True
	for line in open(path_buyer_features):
		if is_first_line:
			is_first_line = False
			continue
		parts = line.split(" ")
		buyer_id = int(parts[0])
		if len(parts)==5:
			gender = int(parts[1])
			age = int(parts[2])
			city = int(parts[3])
			occupation = int(parts[4])
		else:
			gender = ' '
			age = ' '
			city = ' '
			occupation = ' '
		
		for i in range(5):
			if len(parts)==5:
				buyer_feat_set[i].add(int(parts[i]))
			else:
				if i==0:
					buyer_feat_set[i].add(int(parts[i]))
				else:
					buyer_feat_set[i].add(' ')
		
		buyer_feats[buyer_id] = {'gender':gender,'age':age,'city': city,'occupation':occupation}

	for i in range(5):
		def lamb(x):
			if x==' ':
				return -10000
			else:
				return x
		feat_set = list(buyer_feat_set[i])
		feat_set = sorted(feat_set,key=lamb)
		buyer_feat_set[i] = {feat:j+1 for j,feat in enumerate(feat_set)}

	user_feat_batch = {}
	for buyer_id, user_dict in buyer_feats.items():
		user_feature = []
		for key,value in user_dict.items():
			user_feature.append(buyer_feat_set[user_column[key]][value])
		user_feat_batch[buyer_feat_set[user_column['buyer_id']][buyer_id]] = user_feature

	"""
	process trade edge features
	"""
	
	seller_ids = set([])
	trade_records = {}
	user_records = {}
	is_first_line = True
	for line in open(path_edge_features):
		if is_first_line:
			is_first_line = False
			continue
		parts = line.rstrip().split(" ")
		partition = int(parts[0])
		buyer_id = int(parts[1])
		seller_id = int(parts[2])
		trade_create_date = parts[3]
		trade_send_goods_date = parts[4]
		suc_return_date = parts[5]
		trade_total_amt = parts[6]
		trade_return_amt = parts[7]
		dt = int(parts[8])
		if seller_id not in trade_records:
			trade_records[seller_id] = []
		trade_records[seller_id].append({
				'buyer_id': buyer_id,
				'trade_create_date': trade_create_date,
				'trade_send_goods_date': trade_send_goods_date,
				'suc_return_date': suc_return_date,
				'trade_total_amt': trade_total_amt,
				'trade_return_amt': trade_return_amt,
				'dt': dt,
				'partition': partition
				})
		seller_ids.add(seller_id)

		if buyer_id not in user_records:
			user_records[buyer_id] = []
		user_records[buyer_id].append({
				'seller_id': seller_id,
				'trade_create_date': trade_create_date,
				'trade_send_goods_date': trade_send_goods_date,
				'suc_return_date': suc_return_date,
				'trade_total_amt': trade_total_amt,
				'trade_return_amt': trade_return_amt,
				'dt': dt,
				'partition': partition
				})
	
	feat_buy_num = {}
	for buyer_id, buy_num in user_records.items():
		feat_buy_num[buyer_feat_set[user_column['buyer_id']][buyer_id]] = len(buy_num)
	bin_est = KBinsDiscretizer(n_bins=bin_num, encode='ordinal', strategy='quantile')
	bin_est.fit(np.reshape(list(feat_buy_num.values()), [-1,1]))
	for buyer_id, buy_num in feat_buy_num.items():
		feat_buy_num[buyer_id] = bin_est.transform([[buy_num]])[0][0]

	feat_user_trade_amount = {}
	feat_user_avg_trade_amount = {}
	user_avg_trade_amount_vals = []
	for buyer_id, trade_datas in user_records.items():
		buyer_id = buyer_feat_set[user_column['buyer_id']][buyer_id]
		for data in trade_datas:
			trade_total_amt = data['trade_total_amt']
			if trade_total_amt != '':
				if buyer_id not in feat_user_trade_amount:
					feat_user_trade_amount[buyer_id] = []
				trade_total_amt = float(trade_total_amt)
				feat_user_trade_amount[buyer_id].append([trade_total_amt])
	for buyer_id, vals in feat_user_trade_amount.items():
		if buyer_id not in feat_user_avg_trade_amount:
			feat_user_avg_trade_amount[buyer_id] = []
		avg_val = np.mean(vals)
		feat_user_avg_trade_amount[buyer_id] = avg_val
		user_avg_trade_amount_vals.append([avg_val])

	bin_est = KBinsDiscretizer(n_bins=bin_num, encode='ordinal', strategy='quantile')
	bin_est.fit(user_avg_trade_amount_vals)
	for buyer_id, val in feat_user_avg_trade_amount.items():
		feat_user_avg_trade_amount[buyer_id] = bin_est.transform([[val]])[0][0]

	# number of return
	feat_user_return_num = {}
	for buyer_id, trade_datas in user_records.items():
		buyer_id = buyer_feat_set[user_column['buyer_id']][buyer_id]
		return_num = 0
		for data in trade_datas:
			suc_return_date = data['suc_return_date']
			if suc_return_date != '':
				return_num += 1
		feat_user_return_num[buyer_id] = return_num

	return_num_vals = np.array(list(feat_user_return_num.values()))
	return_num_vals = np.reshape(return_num_vals, [-1,1])
	bin_est = KBinsDiscretizer(n_bins=bin_num, encode='ordinal', strategy='quantile')
	bin_est.fit(return_num_vals)
	for buyer_id, val in feat_user_return_num.items():
		bin_res = bin_est.transform([[val]])[0][0]
		feat_user_return_num[buyer_id] = bin_res


	feat_shop_num = {}
	for buyer_id, trade_datas in user_records.items():
		buyer_id = buyer_feat_set[user_column['buyer_id']][buyer_id]
		shop = []
		for record in trade_datas:
			shop.append(record['seller_id'])
		shop_count = Counter(shop)
		feat_shop_num[buyer_id] = len(shop_count)

	bin_est = KBinsDiscretizer(n_bins=bin_num, encode='ordinal', strategy='quantile')
	bin_est.fit(np.reshape(list(feat_shop_num.values()), [-1,1]))
	for buyer_id, val in feat_shop_num.items():
		feat_shop_num[buyer_id] = bin_est.transform([[val]])[0][0]

	feat_user_interval_time = {}
	interval_time_vals = []
	for buyer_id, trade_datas in user_records.items():
		buyer_id = buyer_feat_set[user_column['buyer_id']][buyer_id]
		for data in trade_datas:
			trade_create_date = data['trade_create_date']
			trade_send_goods_date = data['trade_send_goods_date']
			if trade_create_date != '' and trade_send_goods_date != '':
				if buyer_id not in feat_user_interval_time:
					feat_user_interval_time[buyer_id] = []
				trade_create_date = int(trade_create_date)
				trade_send_goods_date = int(trade_send_goods_date)
				interval_time = trade_send_goods_date - trade_create_date
				feat_user_interval_time[buyer_id].append([interval_time])
				interval_time_vals.append([interval_time])
	bin_est = KBinsDiscretizer(n_bins=bin_num, encode='ordinal', strategy='quantile')
	bin_est.fit(interval_time_vals)
	for buyer_id, vals in feat_user_interval_time.items():
		bin_res = bin_est.transform(vals)
		counter_res = Counter(np.reshape(bin_res, [-1]))
		total_count = float(np.sum(list(counter_res.values())))
		histgram_feat = np.zeros([bin_num], dtype=np.float32)
		for i in range(bin_num):
			if i in counter_res:
				histgram_feat[i] = counter_res[i] / total_count
		feat_user_interval_time[buyer_id] = histgram_feat
	
	feature_group = [
		feat_buy_num,
		feat_user_avg_trade_amount,
		feat_user_return_num,
		feat_shop_num,
	]
	
	for uid in user_feat_batch:
		for feature in feature_group:
			if uid in feature:
				user_feat_batch[uid].append(feature[uid])
			else:
				user_feat_batch[uid].append(bin_num)
	
	for uid in user_feat_batch:
		if uid in feat_user_interval_time:
			user_feat_batch[uid].extend(feat_user_interval_time[uid])
		else:
			user_feat_batch[uid].extend([0]*bin_num)
	

	# interval between the time of goods delivery and trade created time
	feat_interval_time = {}
	interval_time_vals = []
	for seller_id, trade_datas in trade_records.items():
		for data in trade_datas:
			trade_create_date = data['trade_create_date']
			trade_send_goods_date = data['trade_send_goods_date']
			if trade_create_date != '' and trade_send_goods_date != '':
				if seller_id not in feat_interval_time:
					feat_interval_time[seller_id] = []
				trade_create_date = int(trade_create_date)
				trade_send_goods_date = int(trade_send_goods_date)
				interval_time = trade_send_goods_date - trade_create_date
				feat_interval_time[seller_id].append([interval_time])
				interval_time_vals.append([interval_time])
	bin_est = KBinsDiscretizer(n_bins=bin_num, encode='ordinal', strategy='quantile')
	bin_est.fit(interval_time_vals)
	for seller_id, vals in feat_interval_time.items():
		bin_res = bin_est.transform(vals)
		counter_res = Counter(np.reshape(bin_res, [-1]))
		total_count = float(np.sum(list(counter_res.values())))
		histgram_feat = np.zeros([bin_num], dtype=np.float32)
		for i in range(bin_num):
			if i in counter_res:
				histgram_feat[i] = counter_res[i] / total_count
		feat_interval_time[seller_id] = histgram_feat

	# total trade number
	feat_trade_num = {}
	for seller_id, trade_datas in trade_records.items():
		feat_trade_num[seller_id] = len(trade_datas)
	bin_est = KBinsDiscretizer(n_bins=bin_num, encode='ordinal', strategy='quantile')
	bin_est.fit(np.reshape(list(feat_trade_num.values()), [-1,1]))
	for seller_id, val in feat_trade_num.items():
		feat_trade_num[seller_id] = bin_est.transform([[val]])[0][0]

	# average trade amount
	feat_trade_amount = {}
	feat_avg_trade_amount = {}
	avg_trade_amount_vals = []
	for seller_id, trade_datas in trade_records.items():
		for data in trade_datas:
			trade_total_amt = data['trade_total_amt']
			if trade_total_amt != '':
				if seller_id not in feat_trade_amount:
					feat_trade_amount[seller_id] = []
				trade_total_amt = float(trade_total_amt)
				feat_trade_amount[seller_id].append([trade_total_amt])
	for seller_id, vals in feat_trade_amount.items():
		if seller_id not in feat_avg_trade_amount:
			feat_avg_trade_amount[seller_id] = []
		avg_val = np.mean(vals)
		feat_avg_trade_amount[seller_id] = avg_val
		avg_trade_amount_vals.append([avg_val])

	bin_est = KBinsDiscretizer(n_bins=bin_num, encode='ordinal', strategy='quantile')
	bin_est.fit(avg_trade_amount_vals)
	for seller_id, val in feat_avg_trade_amount.items():
		feat_avg_trade_amount[seller_id] = bin_est.transform([[val]])[0][0]

	# number of return
	feat_return_num = {}
	for seller_id, trade_datas in trade_records.items():
		return_num = 0
		for data in trade_datas:
			suc_return_date = data['suc_return_date']
			if suc_return_date != '':
				return_num += 1
		feat_return_num[seller_id] = return_num

	return_num_vals = np.array(list(feat_return_num.values()))
	return_num_vals = np.reshape(return_num_vals, [-1,1])
	bin_est = KBinsDiscretizer(n_bins=bin_num, encode='ordinal', strategy='quantile')
	bin_est.fit(return_num_vals)
	for seller_id, val in feat_return_num.items():
		bin_res = bin_est.transform([[val]])[0][0]
		feat_return_num[seller_id] = bin_res

	#user

	feat_user = {}
	feat_user_mask = {}
	feat_user_num = {}
	user_num = 200
	for seller_id, trade_datas in trade_records.items():
		user = []
		for record in trade_datas:
			user.append(record['buyer_id'])
		user_count = Counter(user)
		user_most = user_count.most_common(user_num)
		user = [buyer_feat_set[user_column['buyer_id']][uid[0]] for uid in user_most]
		mask = [1]*len(user) + [0]*(user_num-len(user))
		user = user + [0]*(user_num-len(user))
		
		feat_user_num[seller_id] = len(user_count)
		feat_user[seller_id] = user
		feat_user_mask[seller_id] = mask

	bin_est = KBinsDiscretizer(n_bins=bin_num, encode='ordinal', strategy='quantile')
	bin_est.fit(np.reshape(list(feat_user_num.values()), [-1,1]))
	for seller_id, val in feat_user_num.items():
		feat_user_num[seller_id] = bin_est.transform([[val]])[0][0]

	ret_dict = {'feat_interval_time':feat_interval_time,
				'feat_trade_num':feat_trade_num,
				'feat_avg_trade_amount':feat_avg_trade_amount,
				'feat_return_num':feat_return_num,
				'feat_user_num':feat_user_num,
				'feat_user':feat_user,
				'feat_user_mask':feat_user_mask,
				'user_feat_batch':user_feat_batch
				}

	return ret_dict




def gen_test_data(s_path_buyer_features, s_path_edge_features, s_path_train_label):  # 开放给用户
	# 训练集和测试集样本特征结构一致，使用s_path_buyer_features和s_path_edge_features中的数据提取特征
	# 我们的示例是简化版：直接获取的训练集预处理特征提前的特征
	ret_dict = train_feat_process(s_path_buyer_features, s_path_edge_features, s_path_train_label)
	feat_interval_time=ret_dict['feat_interval_time']
	feat_trade_num=ret_dict['feat_trade_num']
	feat_avg_trade_amount=ret_dict['feat_avg_trade_amount']
	feat_return_num =ret_dict['feat_return_num']
	feat_user_num=ret_dict['feat_user_num']
	feat_user_mask=ret_dict['feat_user_mask']
	feat_user=ret_dict['feat_user']
	user_feat_batch=ret_dict['user_feat_batch']
	# 这是只是测试集的特征处理部分，不含有label
	# a dataset samples
	valid_feat_user_num = []
	valid_feat_user = []
	valid_feat_user_mask = []
	valid_feat_interval_time = []
	valid_feat_trade_num = []
	valid_feat_avg_trade_amount = []
	valid_feat_return_num = []
	bin_num=10

	user_feature_matrix = np.zeros((len(user_feat_batch)+1,len(user_feat_batch[1])), dtype=np.int32)
	user_feature_matrix[:,4:-bin_num]=bin_num
	for uid,feature in user_feat_batch.items():
		user_feature_matrix[uid]=np.array(feature)


	is_first_line = True
	for line in open(s_path_train_label):
		if is_first_line:
			is_first_line = False
			continue

		parts = line.rstrip().split()
		seller_id = int(parts[0])

		if seller_id in feat_user_num:
			valid_feat_user_num.append(feat_user_num[seller_id])
		else:
			valid_feat_user_num.append(bin_num)

		user_num = 200

		if seller_id in feat_user_mask:
			valid_feat_user_mask.append(feat_user_mask[seller_id])
		else:
			valid_feat_user_mask.append([0]*user_num)

		if seller_id in feat_user:
			valid_feat_user.append(user_feature_matrix[feat_user[seller_id]])
		else:
			valid_feat_user.append(user_feature_matrix[[0]*user_num])
		


		if seller_id in feat_interval_time:
			valid_feat_interval_time.append(feat_interval_time[seller_id])
		else:
			valid_feat_interval_time.append(np.zeros([bin_num], dtype=np.float32))

		if seller_id in feat_trade_num:
			valid_feat_trade_num.append(feat_trade_num[seller_id])
		else:
			valid_feat_trade_num.append(bin_num)

		if seller_id in feat_avg_trade_amount:
			valid_feat_avg_trade_amount.append(feat_avg_trade_amount[seller_id])
		else:
			valid_feat_avg_trade_amount.append(bin_num)

		if seller_id in feat_return_num:
			valid_feat_return_num.append(feat_return_num[seller_id])
		else:
			valid_feat_return_num.append(bin_num)

	valid_feats = [
		np.array(valid_feat_interval_time,dtype=np.float32)
		, np.array(valid_feat_trade_num,dtype=np.float32)
		, np.array(valid_feat_avg_trade_amount,dtype=np.float32)
		, np.array(valid_feat_return_num,dtype=np.float32)
		, np.array(valid_feat_user_num,dtype=np.float32)
		, np.array(valid_feat_user,dtype=np.float32)
		, np.array(valid_feat_user_mask,dtype=np.float32)
	]

	return valid_feats


if __name__=="__main__":
	buyer_path='./data/buyer.txt'
	edge_path='./data/edge.txt'
	label_path='./data/label.txt'
	gen_test_data(buyer_path,edge_path,label_path)
