#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_pretty_metrics(df): 
    
    table_metrics = df.describe().T.reset_index().rename({'index': 'metric'}, axis = 1)

    table_metrics = table_metrics[['metric', 'mean', 'std']].drop(index = [0,1])

    precision_data = table_metrics[table_metrics['metric'].str.contains('precision')]
    recall_data = table_metrics[table_metrics['metric'].str.contains('recall')]
    ndcg_data = table_metrics[table_metrics['metric'].str.contains('ndcg')]

    precision_mean = precision_data['mean']
    recall_mean = recall_data['mean']
    ndcg_mean = ndcg_data['mean']

    precision_k = [int(x.split('@')[-1]) for x in precision_data['metric']]
    recall_k = [int(x.split('@')[-1]) for x in recall_data['metric']]
    ndcg_k = [int(x.split('@')[-1]) for x in ndcg_data['metric']]
    
    plt.figure(figsize=(15, 5)) 

    plt.subplot(1, 3, 1)
    plt.plot(precision_k, precision_mean, marker='o', color='black', linewidth=2, markersize=5)
    plt.title('Precision@K', color='black')
    plt.xlabel('K', color='black')
    plt.ylabel('Precision', color='black')
    plt.grid(True, axis='y', linestyle='--', linewidth=0.5, color='gray', alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(recall_k, recall_mean, marker='o', color='black', linewidth=2, markersize=5)
    plt.title('Recall@K', color='black')
    plt.xlabel('K', color='black')
    plt.ylabel('Recall', color='black')
    plt.grid(True, axis='y', linestyle='--', linewidth=0.5, color='gray', alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(ndcg_k, ndcg_mean, marker='o', color='black', linewidth=2, markersize=5)
    plt.title('NDCG@K', color='black')
    plt.xlabel('K', color='black')
    plt.ylabel('NDCG', color='black')
    plt.grid(True, axis='y', linestyle='--', linewidth=0.5, color='gray', alpha=0.3)

    plt.tight_layout()
    plt.show()

