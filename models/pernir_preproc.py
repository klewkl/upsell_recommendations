#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import random
import numpy as np
from scipy import sparse
import similaripy as sim
import pickle
import random

## Raw Data Preprocessing ##

def read_and_merge_orders(prior_orders_file_path, train_orders_file_path, orders_file_path):
    prior_orders = pd.read_csv(prior_orders_file_path)
    train_orders = pd.read_csv(train_orders_file_path)
    all_orders = pd.concat([prior_orders,train_orders])

    order_info = pd.read_csv(orders_file_path)

    all_orders = pd.merge(order_info,all_orders,how='inner')
    all_orders = all_orders.rename(columns={'order_id':'basket_id', 'product_id':'item_id'})

    return all_orders

def filter_orders(all_orders, min_basket_per_user, min_item_per_basket):
    item_per_basket = all_orders[['item_id','basket_id']].drop_duplicates() \
        .groupby('basket_id').agg({'item_id':'count'}).reset_index()

    item_per_basket = item_per_basket[item_per_basket['item_id'] >= min_item_per_basket]
    baskets = set(item_per_basket['basket_id'].tolist())

    all_orders = all_orders[all_orders['basket_id'].isin(baskets)]
    basket_per_user = all_orders[['user_id','basket_id']].drop_duplicates() \
        .groupby('user_id').agg({'basket_id':'count'}).reset_index()

    basket_per_user = basket_per_user[basket_per_user['basket_id'] >= min_basket_per_user]

    return all_orders, basket_per_user

def sample_users(basket_per_user, sample_size):
    all_users = set(basket_per_user['user_id'].tolist())
    users = random.sample(list(all_users),sample_size)
    return users

def split_baskets(all_orders, users):
    all_orders = all_orders[all_orders['user_id'].isin(users)]

    last_baskets = all_orders[['user_id','basket_id','order_number']].drop_duplicates() \
        .groupby('user_id').apply(lambda grp: grp.nlargest(1, 'order_number'))

    last_baskets.index = last_baskets.index.droplevel()
    test_baskets = pd.merge(last_baskets, all_orders, how='left')
    train_baskets = pd.concat([all_orders,test_baskets]).drop_duplicates(keep=False)

    last_baskets = train_baskets[['user_id','basket_id','order_number']].drop_duplicates() \
        .groupby('user_id').apply(lambda grp: grp.nlargest(1, 'order_number'))
    last_baskets.index = last_baskets.index.droplevel()
    valid_baskets = pd.merge(last_baskets, all_orders, how='left')
    train_baskets = pd.concat([train_baskets,valid_baskets]).drop_duplicates(keep=False)

    return train_baskets, test_baskets, valid_baskets

def save_baskets(train_baskets, test_baskets, valid_baskets, train_baskets_file_path, test_baskets_file_path, valid_baskets_file_path):
    train_baskets.to_csv(train_baskets_file_path, index=False)
    test_baskets.to_csv(test_baskets_file_path, index=False)
    valid_baskets.to_csv(valid_baskets_file_path, index=False)
    
    
### Sim Matrix ## 

def create_basket_items_dict(train_baskets):
    baskets_df = train_baskets[['basket_id', 'item_id', 'add_to_cart_order']].drop_duplicates()
    basket_items = baskets_df.sort_values(['basket_id', 'add_to_cart_order']).groupby(['basket_id'])['item_id'] \
        .apply(list).reset_index(name='items')
    return dict(zip(basket_items['basket_id'],basket_items['items']))

def create_user_baskets_dict(train_baskets):
    user_baskets_df = train_baskets[['basket_id','user_id']].drop_duplicates()
    user_baskets = user_baskets_df.groupby(['user_id'])['basket_id'].apply(list) \
        .reset_index(name='baskets')
    return dict(zip(user_baskets['user_id'],user_baskets['baskets']))

def create_item_base_scores(user_baskets_dict, basket_items_dict):
    item_base_scores = {}
    for user in user_baskets_dict:
        baskets = user_baskets_dict[user]
        basket_len = len(baskets)
        if user not in item_base_scores:
            item_base_scores[user] = {}
            for basket_index,basket in enumerate(baskets):
                w1_b = 1./float(basket_len - basket_index)
                for item in basket_items_dict[basket]:
                    if item not in item_base_scores[user]:
                        item_base_scores[user][item] = 0
                    item_base_scores[user][item] += w1_b
    return item_base_scores

def create_data_list(item_base_scores, user_baskets_dict):
    data_list = []
    for user in item_base_scores:
        baskets = user_baskets_dict[user]
        basket_len = len(baskets)
        for item in item_base_scores[user]:
            score = float(item_base_scores[user][item]) / float(basket_len)
            data_list.append([user, item, score])
    return data_list

def save_dataframe(data_list, file_path):
    df = pd.DataFrame(data_list, columns = ['user', 'item','score'])
    df.to_csv(file_path,index=False)

def create_user_item_dicts(df):
    df_users = set(df['user'].tolist())
    df_items = set(df['item'].tolist())
    item_dic = {item: i for i, item in enumerate(df_items)}
    rev_item_dic = {i: item for i, item in enumerate(df_items)}
    user_dic = {user: i for i, user in enumerate(df_users)}
    rev_user_dic = {i: user for i, user in enumerate(df_users)}
    return item_dic, rev_item_dic, user_dic, rev_user_dic

def create_user_item_matrix(df, user_dic, item_dic):
    df['uid'] = df['user'].apply(lambda x: user_dic[x])
    df['pid'] = df['item'].apply(lambda x: item_dic[x])
    n_users = len(set(df['user'].tolist()))
    n_items = len(set(df['item'].tolist()))
    userItem_mat = sparse.coo_matrix(( df.score.values  , (df.uid.values, df.pid.values)), shape=(n_users,n_items))
    return userItem_mat

def create_user_sim_dict(userItem_mat, user_dic, rev_user_dic):
    userSim = sim.asymmetric_cosine(sparse.csr_matrix(userItem_mat), alpha=0.5, k=50)
    user_sim_dict = dict(userSim.todok().items())
    final_user_sim_dict = {}
    for key in user_sim_dict:
        final_user_sim_dict[(rev_user_dic[key[0]],rev_user_dic[key[1]])] = user_sim_dict[key]
    return final_user_sim_dict

def save_user_sim_dict(final_user_sim_dict, file_path):
    with open(file_path, 'wb') as handle:
        pickle.dump(final_user_sim_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
## Create test sample ###


def read_and_process_baskets(basket_path):
    baskets_df = pd.read_csv(basket_path)
    basket_users = baskets_df[['basket_id', 'user_id']].drop_duplicates()
    basket_users_dict = dict(zip(basket_users['basket_id'], basket_users['user_id']))

    baskets_df = baskets_df[['basket_id', 'item_id', 'add_to_cart_order']].drop_duplicates()
    basket_items = baskets_df.sort_values(['basket_id', 'add_to_cart_order']).groupby(['basket_id'])['item_id'] \
        .apply(list).reset_index(name='items')
    basket_items_dict = dict(zip(basket_items['basket_id'], basket_items['items']))

    return basket_users_dict, basket_items_dict

def create_masked_baskets(basket_items_dict, basket_users_dict, mask_percentage):
    sample_baskets = []
    sample_users = []
    full_baskets = []
    masked_items_list = []
    masked_baskets = []
    num_to_mask_list = []

    for basket in basket_items_dict:
        items = basket_items_dict[basket]
        num_items_to_mask = max(1, int(len(items) * mask_percentage))
        mask_indices = np.random.choice(len(items), num_items_to_mask, replace=False)

        masked_items = [items[i] for i in mask_indices]
        masked_basket = [item for i, item in enumerate(items) if i not in mask_indices]

        sample_baskets.append(basket)
        sample_users.append(basket_users_dict[basket])
        full_baskets.append(items)
        masked_items_list.append(masked_items)
        masked_baskets.append(masked_basket)
        num_to_mask_list.append(num_items_to_mask)

    return sample_baskets, sample_users, full_baskets, masked_items_list, masked_baskets, num_to_mask_list

def create_sample_dataframe(sample_baskets, sample_users, full_baskets, masked_items_list, masked_baskets, num_to_mask_list):
    sample_df = pd.DataFrame({
        'basket_id': sample_baskets,
        'user_id': sample_users,
        'full_basket': full_baskets,
        'masked_items': masked_items_list,
        'masked_basket': masked_baskets,
        'num_to_mask': num_to_mask_list
    })
    return sample_df

