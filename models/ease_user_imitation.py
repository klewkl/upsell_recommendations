#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import scipy.sparse as sps
from tqdm import tqdm

def create_index_mappings(data, user_col, item_col):
    user2idx = {v: k for k, v in enumerate(data[user_col].unique())}
    item2idx = {v: k for k, v in enumerate(data[item_col].unique())}
    idx2user = {k: v for v, k in user2idx.items()}
    idx2item = {k: v for v, k in item2idx.items()}

    data[user_col] = data[user_col].apply(lambda x: user2idx[x])
    data[item_col] = data[item_col].apply(lambda x: item2idx[x])

    return user2idx, item2idx, idx2user, idx2item, data

def create_sparse_matrix(df, order_columns, item_column, user2idx, item2idx):
    matrix = sps.coo_matrix(
        (np.ones(df.shape[0]), (df[order_columns], df[item_column])),
        shape=(len(user2idx), len(item2idx))
    )
    return matrix

def fit_ease(X, reg_weight=100):
    G = X.T @ X
    G += reg_weight * sps.identity(G.shape[0])
    G = G.todense()
    P = np.linalg.inv(G)
    B = P / (-np.diag(P))
    np.fill_diagonal(B, 0.)
    return B

def get_user_vector(basket, item2idx):
    vector = np.zeros(len(item2idx))
    for item in basket:
        if item in item2idx:
            vector[item2idx[item]] = 1
    return vector

def generate_recommendations(pred, w):
    
    """
    Generating item recommendations for each user based on their user vector and the item-item similarity matrix.

    Args:
        pred (pd.DataFrame): A DataFrame containing user vectors in the 'vector' column.
        item2idx (dict): A dictionary mapping item values to integer indices.
        w (numpy.ndarray): A dense matrix representing the item-item similarity scores.

    Returns:
        recs_for_user (list): A list of lists containing the top recommended items for each user.
    """
    scores = []
    recs_for_user = []
    batch = []

    for idx, row in tqdm(pred.iterrows(), total=pred.shape[0]):
        vector = row.vector
        batch.append(vector)

        if len(batch) > 10000:
            batch = np.array(batch)
            user_scores = batch.dot(w)
            user_scores = np.argsort(-user_scores)[:, :200]
            for i in range(len(user_scores)):
                recs_for_user.append(user_scores[i].tolist())
            batch = []

    batch = np.array(batch)
    user_scores = batch.dot(w)
    user_scores = np.argsort(-user_scores)[:, :200]
    for i in range(len(user_scores)):
        recs_for_user.append(user_scores[i].tolist())

    return recs_for_user

def get_decoded_recommendations(x, item2idx, idx2item):
    
    
    recs = []
    consumed = [item2idx[t] for t in x.full_basket if t in item2idx]
    for el in x.recs[0]:
        recs.append(idx2item[el])
        if len(recs) == 20:
            break
            
    return recs

def recommend_items(train_data, test_data, user_col, item_col, top_n=20):
    print("Creating index mappings...")
    user2idx, item2idx, idx2user, idx2item, train_data = create_index_mappings(train_data, user_col, item_col)
    print("Index mappings created.")

    print("Creating sparse matrix...")
    matrix = create_sparse_matrix(train_data, user_col, item_col, user2idx, item2idx)
    print("Sparse matrix created.")

    print("Fitting EASE...")
    w = fit_ease(matrix)
    print("EASE fitted.")

    print("Generating user vectors...")
    test_data['vector'] = test_data.masked_basket.apply(lambda x: get_user_vector(x, item2idx))
    print("User vectors generated.")

    print("Generating recommendations...")
    recs_for_user = generate_recommendations(test_data, w)
    test_data['recs'] = recs_for_user
    test_data['preds'] = test_data.apply(lambda x: get_decoded_recommendations(x, item2idx, idx2item), axis=1)
    print("Recommendations generated.")

    return test_data
