import sys
import os
import pickle
import sklearn
import numpy as np
from collections import Counter
from sklearn.preprocessing import KBinsDiscretizer
import torch
import json
import os
import numpy as np
import wget
import zipfile
import logging
import warnings
warnings.filterwarnings("ignore")

# training samples
def gen_train_data(c_path_buyer_features, c_path_edge_features, c_path_train_label):
	ret_dict = train_feat_process(c_path_buyer_features, c_path_edge_features, c_path_train_label)
	feat_interval_time=ret_dict['feat_interval_time']
	seller_feat=ret_dict['seller_feat']
	feat_user=ret_dict['feat_user']
	user_feat_batch=ret_dict['user_feat_batch']
	feat_user_buy_num=ret_dict['feat_user_buy_num']
	feat_user_buy_amount=ret_dict['feat_user_buy_amount']
	feat_user_return_amount=ret_dict['feat_user_return_amount']


	c_path_train_label = c_path_train_label
	train_feat_user_buy_amount = []
	train_feat_user = []
	train_feat_user_buy_num=[]
	train_feat_interval_time = []
	train_feat_seller = []
	train_feat_user_return_amount=[]

	train_labels = []
	bin_num=15
	num_time_feat=2
	seller_feat_num = len(next(iter(seller_feat.values())))

	user_feature_matrix = np.zeros((len(user_feat_batch)+1,len(user_feat_batch[1])), dtype=np.int32)
	user_feature_matrix[:,4:-bin_num*num_time_feat]=bin_num
	for uid,feature in user_feat_batch.items():
		user_feature_matrix[uid]=np.array(feature)

	is_first_line = True
	for line in open(c_path_train_label):
		if is_first_line:
			is_first_line = False
			continue

		parts = line.rstrip().split()
		seller_id = int(parts[0])
		label = float(parts[1])
		# sample_partition = int(parts[2])

		#if partition is not None and sample_partition != partition:
		#	continue

		train_labels.append(label)

		user_num = 200

		if seller_id in feat_user_return_amount:
			train_feat_user_return_amount.append(feat_user_return_amount[seller_id]+[bin_num]*(user_num-len(feat_user_return_amount[seller_id])))
		else:
			train_feat_user_return_amount.append([bin_num]*user_num)

		if seller_id in feat_user_buy_amount:
			train_feat_user_buy_amount.append(feat_user_buy_amount[seller_id]+[bin_num]*(user_num-len(feat_user_buy_amount[seller_id])))
		else:
			train_feat_user_buy_amount.append([bin_num]*user_num)

		if seller_id in feat_user_buy_num:
			train_feat_user_buy_num.append(feat_user_buy_num[seller_id]+[bin_num]*(user_num-len(feat_user_buy_num[seller_id])))
		else:
			train_feat_user_buy_num.append([bin_num]*user_num)

		if seller_id in feat_user:
			train_feat_user.append(user_feature_matrix[feat_user[seller_id]])
		else:
			train_feat_user.append(user_feature_matrix[[0]*user_num])

		if seller_id in feat_interval_time:
			train_feat_interval_time.append(feat_interval_time[seller_id])
		else:
			train_feat_interval_time.append(np.zeros([bin_num*num_time_feat], dtype=np.float32))
		
		if seller_id in seller_feat:
			train_feat_seller.append(seller_feat[seller_id])
		else:
			train_feat_seller.append([bin_num]*seller_feat_num)


	indices = np.arange(len(train_labels))
	indices = np.random.permutation(indices)

	train_feats = [
		np.array(train_feat_interval_time,dtype=np.float32)[indices]
		, np.array(train_feat_seller,dtype=np.float32)[indices]
		, np.array(train_feat_user,dtype=np.float32)[indices]
		, np.array(train_feat_user_buy_num,dtype=np.float32)[indices]
		, np.array(train_feat_user_buy_amount,dtype=np.float32)[indices]
		, np.array(train_feat_user_return_amount,dtype=np.float32)[indices]
		]
	train_labels = np.array(train_labels)[indices]
	train_data = []
	train_data.append(train_feats)
	train_data.append(train_labels)

	return train_data, len(train_labels)

def train_feat_process(path_buyer_features, path_edge_features, path_train_label): # 开放给用户
	
	"""
	process trade buyer features
	"""
	bin_num=15
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

	feat_user_return_amount = {}
	feat_user_avg_return_amount = {}
	user_avg_return_amount_vals = []
	for buyer_id, trade_datas in user_records.items():
		buyer_id = buyer_feat_set[user_column['buyer_id']][buyer_id]
		for data in trade_datas:
			return_total_amt = data['trade_return_amt']
			if return_total_amt == '':
				return_total_amt='0'
			if buyer_id not in feat_user_return_amount:
				feat_user_return_amount[buyer_id] = []
			return_total_amt = float(return_total_amt)
			feat_user_return_amount[buyer_id].append([return_total_amt])
			
	for buyer_id, vals in feat_user_return_amount.items():
		if buyer_id not in feat_user_avg_return_amount:
			feat_user_avg_return_amount[buyer_id] = []
		avg_val = np.mean(vals)
		feat_user_avg_return_amount[buyer_id] = avg_val
		user_avg_return_amount_vals.append([avg_val])

	bin_est = KBinsDiscretizer(n_bins=bin_num, encode='ordinal', strategy='quantile')
	bin_est.fit(user_avg_return_amount_vals)
	for buyer_id, val in feat_user_avg_return_amount.items():
		feat_user_avg_return_amount[buyer_id] = bin_est.transform([[val]])[0][0]
	
	feat_user_profit_amount = {}
	feat_user_avg_profit_amount = {}
	user_avg_profit_amount_vals = []
	for buyer_id, trade_datas in user_records.items():
		buyer_id = buyer_feat_set[user_column['buyer_id']][buyer_id]
		for data in trade_datas:
			return_total_amt = data['trade_return_amt']
			trade_total_amt = data['trade_total_amt']
			if trade_total_amt != '':
				if return_total_amt == '':
					return_total_amt='0'
				if buyer_id not in feat_user_profit_amount:
					feat_user_profit_amount[buyer_id] = []
				return_total_amt = float(return_total_amt)
				trade_total_amt = float(trade_total_amt)
				profit_amt = trade_total_amt - return_total_amt
				feat_user_profit_amount[buyer_id].append([profit_amt])
			
	for buyer_id, vals in feat_user_profit_amount.items():
		if buyer_id not in feat_user_avg_profit_amount:
			feat_user_avg_profit_amount[buyer_id] = []
		avg_val = np.mean(vals)
		feat_user_avg_profit_amount[buyer_id] = avg_val
		user_avg_profit_amount_vals.append([avg_val])

	bin_est = KBinsDiscretizer(n_bins=bin_num, encode='ordinal', strategy='quantile')
	bin_est.fit(user_avg_profit_amount_vals)
	for buyer_id, val in feat_user_avg_profit_amount.items():
		feat_user_avg_profit_amount[buyer_id] = bin_est.transform([[val]])[0][0]

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
	
	feat_user_return_interval_time = {}
	user_return_interval_time_vals = []
	for buyer_id, trade_datas in user_records.items():
		buyer_id = buyer_feat_set[user_column['buyer_id']][buyer_id]
		for data in trade_datas:
			suc_return_date = data['suc_return_date']
			trade_create_date = data['trade_create_date']
			if suc_return_date != '' and trade_create_date != '':
				if buyer_id not in feat_user_return_interval_time:
					feat_user_return_interval_time[buyer_id] = []
				trade_create_date = int(trade_create_date)
				suc_return_date = int(suc_return_date)
				interval_time = suc_return_date - trade_create_date
				feat_user_return_interval_time[buyer_id].append([interval_time])
				user_return_interval_time_vals.append([interval_time])
	bin_est = KBinsDiscretizer(n_bins=bin_num, encode='ordinal', strategy='quantile')
	bin_est.fit(user_return_interval_time_vals)
	for buyer_id, vals in feat_user_return_interval_time.items():
		bin_res = bin_est.transform(vals)
		counter_res = Counter(np.reshape(bin_res, [-1]))
		total_count = float(np.sum(list(counter_res.values())))
		histgram_feat = np.zeros([bin_num], dtype=np.float32)
		for i in range(bin_num):
			if i in counter_res:
				if total_count==0:
					histgram_feat[i] = 0
				else:
					histgram_feat[i] = counter_res[i] / total_count
		feat_user_return_interval_time[buyer_id] = histgram_feat
	
	feature_group = [
		feat_buy_num,
		feat_user_avg_trade_amount,
		feat_user_avg_return_amount,
		feat_user_avg_profit_amount,
		feat_user_return_num,
		feat_shop_num,
	]
	
	for uid in user_feat_batch:
		for feature in feature_group:
			if uid in feature:
				user_feat_batch[uid].append(feature[uid])
			else:
				user_feat_batch[uid].append(bin_num)
	
	feat_time=[
		feat_user_interval_time,
		feat_user_return_interval_time
	]

	for uid in user_feat_batch:
		for feat in feat_time:
			if uid in feat:
				user_feat_batch[uid].extend(feat[uid])
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
	
	# interval between the time of goods delivery and trade return time
	feat_return_interval_time = {}
	return_interval_time_vals = []
	for seller_id, trade_datas in trade_records.items():
		for data in trade_datas:
			suc_return_date = data['suc_return_date']
			trade_create_date = data['trade_create_date']
			if suc_return_date != '' and trade_create_date != '':
				if seller_id not in feat_return_interval_time:
					feat_return_interval_time[seller_id] = []
				trade_create_date = int(trade_create_date)
				suc_return_date = int(suc_return_date)
				interval_time = suc_return_date - trade_create_date
				feat_return_interval_time[seller_id].append([interval_time])
				return_interval_time_vals.append([interval_time])
	bin_est = KBinsDiscretizer(n_bins=bin_num, encode='ordinal', strategy='quantile')
	bin_est.fit(return_interval_time_vals)
	for seller_id, vals in feat_return_interval_time.items():
		bin_res = bin_est.transform(vals)
		counter_res = Counter(np.reshape(bin_res, [-1]))
		total_count = float(np.sum(list(counter_res.values())))
		histgram_feat = np.zeros([bin_num], dtype=np.float32)
		for i in range(bin_num):
			if i in counter_res:
				if total_count==0:
					histgram_feat[i] = 0
				else:
					histgram_feat[i] = counter_res[i] / total_count
		feat_return_interval_time[seller_id] = histgram_feat

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
	
	feat_return_amount = {}
	feat_avg_return_amount = {}
	avg_return_amount_vals = []
	for seller_id, trade_datas in trade_records.items():
		for data in trade_datas:
			trade_return_amt = data['trade_return_amt']
			if trade_return_amt == '':
				trade_return_amt='0'
			if seller_id not in feat_return_amount:
				feat_return_amount[seller_id] = []
			trade_return_amt = float(trade_return_amt)
			feat_return_amount[seller_id].append([trade_return_amt])
	for seller_id, vals in feat_return_amount.items():
		if seller_id not in feat_avg_return_amount:
			feat_avg_return_amount[seller_id] = []
		avg_val = np.mean(vals)
		feat_avg_return_amount[seller_id] = avg_val
		avg_return_amount_vals.append([avg_val])

	bin_est = KBinsDiscretizer(n_bins=bin_num, encode='ordinal', strategy='quantile')
	bin_est.fit(avg_return_amount_vals)
	for seller_id, val in feat_avg_return_amount.items():
		feat_avg_return_amount[seller_id] = bin_est.transform([[val]])[0][0]

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

	all_buy_num = []
	all_buy_amount = []
	all_return_amount = []
	feat_user = {}
	feat_user_num = {}
	feat_user_buy_num={}
	feat_user_buy_amount={}
	feat_user_return_amount={}
	user_num = 200
	for seller_id, trade_datas in trade_records.items():
		user = []
		user_buy_amount = {}
		user_return_amount = {}
		for record in trade_datas:
			user.append(record['buyer_id'])
			buyer_id = buyer_feat_set[user_column['buyer_id']][record['buyer_id']]
			trade_total_amt = record['trade_total_amt']
			trade_return_amt = record['trade_return_amt']
			if trade_total_amt != '':
				if buyer_id not in user_buy_amount:
					user_buy_amount[buyer_id] = [float(trade_total_amt)]
				else:
					user_buy_amount[buyer_id].append(float(trade_total_amt))
			if trade_return_amt == '':
				trade_return_amt='0'                           
			if buyer_id not in user_return_amount:
				user_return_amount[buyer_id] = [float(trade_return_amt)]
			else:
				user_return_amount[buyer_id].append(float(trade_return_amt))
		for key in user_buy_amount:
			user_buy_amount[key] = np.mean(user_buy_amount[key])
		for key in user_return_amount:
			user_return_amount[key] = np.mean(user_return_amount[key])

		all_buy_amount.extend(list(user_buy_amount.values()))
		all_return_amount.extend(list(user_return_amount.values()))
		user_count = Counter(user)
		user_most = user_count.most_common(user_num)
		user = [buyer_feat_set[user_column['buyer_id']][uid[0]] for uid in user_most]
		padding_num = (user_num-len(user))
		user = user + [0]*padding_num

		user_buy_num = [uid[1] for uid in user_most] 
		all_buy_num.extend(user_buy_num)

		feat_user_buy_num[seller_id] = user_buy_num
		feat_user_num[seller_id] = len(user_count)
		feat_user[seller_id] = user
		user_buy_amount = [user_buy_amount[buyer_feat_set[user_column['buyer_id']][uid[0]]] for uid in user_most]
		user_return_amount = [user_return_amount[buyer_feat_set[user_column['buyer_id']][uid[0]]] for uid in user_most]
		feat_user_buy_amount[seller_id] = user_buy_amount
		feat_user_return_amount[seller_id] = user_return_amount
	
	bin_est = KBinsDiscretizer(n_bins=bin_num, encode='ordinal', strategy='quantile')
	bin_est.fit(np.reshape(all_buy_num, [-1,1]))
	for seller_id,val in feat_user_buy_num.items():
		temp = []
		for v in val:
			temp.append(bin_est.transform([[v]])[0][0])
		feat_user_buy_num[seller_id] = temp
	
	bin_est = KBinsDiscretizer(n_bins=bin_num, encode='ordinal', strategy='quantile')
	bin_est.fit(np.reshape(all_buy_amount, [-1,1]))
	for seller_id,val in feat_user_buy_amount.items():
		temp = []
		for v in val:
			temp.append(bin_est.transform([[v]])[0][0])
		feat_user_buy_amount[seller_id] = temp
	
	bin_est = KBinsDiscretizer(n_bins=bin_num, encode='ordinal', strategy='quantile')
	bin_est.fit(np.reshape(all_return_amount, [-1,1]))
	for seller_id,val in feat_user_return_amount.items():
		temp = []
		for v in val:
			temp.append(bin_est.transform([[v]])[0][0])
		feat_user_return_amount[seller_id] = temp

	bin_est = KBinsDiscretizer(n_bins=bin_num, encode='ordinal', strategy='quantile')
	bin_est.fit(np.reshape(list(feat_user_num.values()), [-1,1]))
	for seller_id, val in feat_user_num.items():
		feat_user_num[seller_id] = bin_est.transform([[val]])[0][0]
	
	seller_feat={}
	feat_group = [
				feat_trade_num,
				feat_avg_trade_amount,
				feat_return_num,
				feat_user_num,
				feat_avg_return_amount
	]

	for seller_id in seller_ids:
		seller_feat[seller_id]=[]
		for feat in feat_group:
			if seller_id in feat:
				seller_feat[seller_id].append(feat[seller_id])
			else:
				seller_feat[seller_id].append(bin_num)
	
	feat_group = [
		feat_interval_time,
		feat_return_interval_time
	]

	feat_time = {}
	for seller_id in seller_ids:
		feat_time[seller_id]=[]
		for feat in feat_group:
			if seller_id in feat:
				feat_time[seller_id].extend(feat[seller_id])
			else:
				feat_time[seller_id].extend([0]*bin_num)

	ret_dict = {'feat_interval_time':feat_time,
				'seller_feat':seller_feat,
				'feat_user':feat_user,
				'feat_user_buy_num':feat_user_buy_num,
				'feat_user_buy_amount':feat_user_buy_amount,
				'user_feat_batch':user_feat_batch,
				'feat_user_return_amount':feat_user_return_amount
				}

	return ret_dict


if __name__=="__main__":
	buyer_path='./data/buyer.txt'
	edge_path='./data/edge.txt'
	label_path='./data/label.txt'
	gen_train_data(buyer_path,edge_path,label_path)