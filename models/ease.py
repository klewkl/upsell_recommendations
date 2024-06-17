#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
from tqdm import tqdm


def create_index_mappings(data, user_col, item_col):
    
    """  
    Creates index mappings for unique values in user_col and item_col, 
    and update the DataFrame with new integer-based indices.

    """
    user2idx = {v: k for k, v in enumerate(data[user_col].unique())}
    item2idx = {v: k for k, v in enumerate(data[item_col].unique())}
    idx2user = {k:v for v, k in user2idx.items()}
    idx2item = {k:v for v, k in item2idx.items()}

    data[user_col] = data[user_col].apply(lambda x: user2idx[x])
    data[item_col] = data[item_col].apply(lambda x: item2idx[x])

    return user2idx, item2idx, idx2user, idx2item, data


def create_sparce_matrix(df, order_columns, item_column, user2idx, item2idx): 
    
    """
    Creates a sparse matrix representing the interactions between orders and items
    
    Args:
        df (pd.DataFrame): The input DataFrame containing orders and items.
        order_columns (str or list of str): The name(s) of the order column(s) in the DataFrame.
        item_column (str): The name of the item column in the DataFrame.
        user2idx (dict): A dictionary mapping user values to integer indices.
        item2idx (dict): A dictionary mapping item values to integer indices.

    Returns:
        matrix (scipy.sparse.coo_matrix): A sparse matrix with interactions between orders and items
    
    """
    
    matrix = sps.coo_matrix(
    (np.ones(df.shape[0]), (df[order_columns], df[item_column])),
    shape=(len(user2idx), len(item2idx)))
    
    return matrix


def fit_ease(X, reg_weight=100):
    
    """
    Fit the EASE (Embarrassingly Shallow AutoEncoder, EASE in reverse) algorithm to a sparse matrix X.

    Args:
        X (scipy.sparse.coo_matrix): A sparse matrix representing the interactions between orders and items.
        reg_weight (float, optional): The regularization weight. Defaults to 100.

    Returns:
        B (numpy.ndarray): A dense matrix representing the item-item similarity scores.
    """
    
    G = X.T @ X
    G += reg_weight * sps.identity(G.shape[0])
    G = G.todense()
    P = np.linalg.inv(G)
    B = P / (-np.diag(P))
    np.fill_diagonal(B, 0.)
    
    return B


def get_user_vector(basket, item2idx):
    
    """
    Create a binary user vector based on the items in the basket and the item2idx mapping.

    Args:
        basket (list): A list of items purchased by a user.
        item2idx (dict): A dictionary mapping item values to integer indices.

    Returns:
        vector (numpy.ndarray): A binary vector representing the items in the basket.
    """
    
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
    
    """
    Decoding the recommended item indices to their original item values.

    Args:
        x (pd.Series): A row from a DataFrame containing 'basket' and 'recs' columns.
        item2idx (dict): A dictionary mapping item values to integer indices.
        idx2item (dict): A dictionary mapping integer indices to item values.

    Returns:
        recs (list): A list of the top 20 recommended items in their original item values.
    """
    
    recs = []
    consumed = [item2idx[t] for t in x.basket if t in item2idx]
    for el in x.recs[0]:
        recs.append(idx2item[el])
        if len(recs) == 20:
            break
            
    return recs


def recommend_items(train_data, test_data, user_col, item_col, top_n=20):
    """
    Generate top-N item recommendations for each user using the EASE algorithm.

    Args:
        data (pd.DataFrame): The input DataFrame containing user and item columns.
        user_col (str): The name of the user column.
        item_col (str): The name of the item column.
        top_n (int, optional): The number of top recommendations to generate. Defaults to 200.

    Returns:
        pred (pd.DataFrame): A DataFrame containing the recommended items for each user.
    """
    print("Creating index mappings...")
    user2idx, item2idx, idx2user, idx2item, train_data = create_index_mappings(train_data, user_col, item_col)
    print("Index mappings created.")

    print("Creating sparse matrix...")
    matrix = create_sparce_matrix(train_data, user_col, item_col, user2idx, item2idx)
    print("Sparse matrix created.")

    print("Fitting EASE...")
    w = fit_ease(matrix)
    print("EASE fitted.")

    print("Generating user vectors...")
    pred = test_data.groupby([user_col])[item_col].agg([('basket', list)]).reset_index()
    pred['vector'] = pred.basket.apply(lambda x: get_user_vector(x, item2idx))
    print("User vectors generated.")

    print("Generating recommendations...")
    recs_for_user = generate_recommendations(pred, w)
    pred['recs'] = recs_for_user
    pred['preds'] = pred.apply(lambda x: get_decoded_recommendations(x, item2idx, idx2item), axis=1)
    print("Recommendations generated.")

    return pred

