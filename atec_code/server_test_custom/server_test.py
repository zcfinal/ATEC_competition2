import json
import os
import numpy as np
import torch
from collections import Counter
from sklearn.preprocessing import KBinsDiscretizer

def train_feat_process(path_buyer_features, path_edge_features, path_train_label):

	
	buyer_feats = {}
	is_first_line = True
	for line in open(path_buyer_features):
		if is_first_line:
			is_first_line = False
			continue
		parts = line.rstrip().split()
		buyer_id = int(parts[0])
		city = int(parts[3])
		buyer_feats[buyer_id] = {'city': city}

	
	seller_ids = set([])
	trade_records = {}
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
	bin_est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
	bin_est.fit(interval_time_vals)
	for seller_id, vals in feat_interval_time.items():
		bin_res = bin_est.transform(vals)
		counter_res = Counter(np.reshape(bin_res, [-1]))
		total_count = float(np.sum(list(counter_res.values())))
		histgram_feat = np.zeros([10], dtype=np.float32)
		for i in range(10):
			if i in counter_res:
				histgram_feat[i] = counter_res[i] / total_count
		feat_interval_time[seller_id] = histgram_feat

	# total trade number
	feat_trade_num = {}
	for seller_id, trade_datas in trade_records.items():
		feat_trade_num[seller_id] = len(trade_datas)
	bin_est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
	bin_est.fit(np.reshape(list(feat_trade_num.values()), [-1, 1]))
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

	bin_est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
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
	return_num_vals = np.reshape(return_num_vals, [-1, 1])
	bin_est = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
	bin_est.fit(return_num_vals)
	for seller_id, val in feat_return_num.items():
		bin_res = bin_est.transform([[val]])[0][0]
		feat_return_num[seller_id] = bin_res

	return feat_interval_time, feat_trade_num, feat_avg_trade_amount, feat_return_num

def gen_test_data(s_path_buyer_features, s_path_edge_features, s_path_train_label):  # 开放给用户
	# 训练集和测试集样本特征结构一致，使用s_path_buyer_features和s_path_edge_features中的数据提取特征
	# 我们的示例是简化版：直接获取的训练集预处理特征提前的特征
	feat_interval_time, feat_trade_num, feat_avg_trade_amount, feat_return_num = train_feat_process(s_path_buyer_features, s_path_edge_features, s_path_train_label)
	# 这是只是测试集的特征处理部分，不含有label
	# a dataset samples
	valid_feat_interval_time = []
	valid_feat_trade_num = []
	valid_feat_avg_trade_amount = []
	valid_feat_return_num = []

	is_first_line = True
	for line in open(s_path_train_label):
		if is_first_line:
			is_first_line = False
			continue

		parts = line.rstrip().split()
		seller_id = int(parts[0])

		if seller_id in feat_interval_time:
			valid_feat_interval_time.append(feat_interval_time[seller_id])
		else:
			valid_feat_interval_time.append(np.zeros([10], dtype=np.float32))

		if seller_id in feat_trade_num:
			valid_feat_trade_num.append(feat_trade_num[seller_id])
		else:
			valid_feat_trade_num.append(10)

		if seller_id in feat_avg_trade_amount:
			valid_feat_avg_trade_amount.append(feat_avg_trade_amount[seller_id])
		else:
			valid_feat_avg_trade_amount.append(10)

		if seller_id in feat_return_num:
			valid_feat_return_num.append(feat_return_num[seller_id])
		else:
			valid_feat_return_num.append(10)

	valid_feats = [
		np.array(valid_feat_interval_time)
		, np.array(valid_feat_trade_num)
		, np.array(valid_feat_avg_trade_amount)
		, np.array(valid_feat_return_num)
	]
	return valid_feats


