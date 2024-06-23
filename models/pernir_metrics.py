#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import log

def dcg_at_k(scores, k):
    return sum(score / np.log2(idx + 2) for idx, score in enumerate(scores[:k]))


def idcg_at_k(scores, k):
    sorted_scores = sorted(scores, reverse=True)
    return dcg_at_k(sorted_scores, k)


def ndcg_at_k(original_basket, preds, k=20):
    relevance_scores = [1 if item in original_basket else 0 for item in preds[:k]]
    dcg = dcg_at_k(relevance_scores, k)
    idcg = idcg_at_k(relevance_scores, k)
    return dcg / idcg if idcg > 0 else 0


def calculate_and_save_ndcg(df):
    for index, row in df.iterrows():
        original_basket = eval(row['full_basket']) 
        preds = row['pred'] 

        for k in range(1, 21):
            ndcg = ndcg_at_k(original_basket, preds, k)
            df.at[index, f'ndcg@{k}'] = ndcg
            
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
        original_basket = eval(row['full_basket'])  
        masked_basket = eval(row['masked_basket']) 
        preds = row['pred'] 
        
        precisions, recalls = precision_recall_at_k(masked_basket, preds, original_basket)
        
        for k, (precision, recall) in enumerate(zip(precisions, recalls), start=1):
            df.at[index, f'precision@{k}'] = precision
            df.at[index, f'recall@{k}'] = recall