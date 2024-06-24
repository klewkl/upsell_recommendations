#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import log

def precision_recall_at_k(masked_basket, preds, original_basket):

    precision = []
    recall = []
    masked_and_preds = masked_basket.copy()
    k = 0
    while k < len(preds) and len(masked_and_preds):
        if preds[k] not in masked_and_preds:
            masked_and_preds.append(preds[k]) 
        intersection = len(set(masked_and_preds).intersection(set(original_basket)))
        precision.append(intersection / len(masked_and_preds))
        recall.append(intersection / len(original_basket))
        k += 1
    return precision, recall


def calculate_and_save_precision_recall(df):
    for index, row in df.iterrows():
        original_basket = row['original_basket']
        masked_basket = row['masked_basket']
        preds = row['preds']
        
        precisions, recalls = precision_recall_at_k(masked_basket, preds, original_basket)
        
        for k, (precision, recall) in enumerate(zip(precisions, recalls), start=1):
            df.at[index, f'precision@{k}'] = precision
            df.at[index, f'recall@{k}'] = recall
            
            
def ndcg_at_k(masked_basket, preds, original_basket):
    masked_and_preds = masked_basket.copy()
    gain = [0.0] * len(preds)
    idcg = 0.0
    for k, item in enumerate(preds):
        if item not in masked_and_preds:
            masked_and_preds.append(item)
        if item in original_basket:
            gain[k] = 1 / log(k + 2, 2)  
            idcg += 1 / log(len(set(masked_and_preds).intersection(set(original_basket))) + 1, 2)
    return [gain[k] / idcg if idcg != 0 else 0.0 for k in range(len(preds))]

def calculate_and_save_ndcg(df):
    for index, row in df.iterrows():
        original_basket = row['original_basket']
        masked_basket = row['masked_basket']
        preds = row['preds']

        ndcg = ndcg_at_k(masked_basket, preds, original_basket)

        for k, ndcg_value in enumerate(ndcg, start=1):
            df.at[index, f'ndcg@{k}'] = ndcg_value

            
def calculate_precision_recall_top_n_items(df, top_n_recommendations):
    for index, row in df.iterrows():
        original_basket = row['original_basket']
        masked_basket = row['masked_basket']
        user_id = row['customer_id']
        preds = top_n_recommendations.get(user_id, [])  

        precisions, recalls = precision_recall_at_k(masked_basket, preds, original_basket)

        for k, (precision, recall) in enumerate(zip(precisions, recalls), start=1):
            df.at[index, f'precision@{k}'] = precision
            df.at[index, f'recall@{k}'] = recall

def calculate_and_save_ndcg_top_n_items(df, top_n_recommendations):
    for index, row in df.iterrows():
        original_basket = row['original_basket']
        masked_basket = row['masked_basket']
        user_id = row['customer_id']
        preds = top_n_recommendations.get(user_id, [])  

        ndcg = ndcg_at_k(masked_basket, preds, original_basket)

        for k, ndcg_value in enumerate(ndcg, start=1):
            df.at[index, f'ndcg@{k}'] = ndcg_value