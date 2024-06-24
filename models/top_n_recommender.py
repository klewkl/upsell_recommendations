#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from collections import Counter

class TopNRecommender:
    def __init__(self, dataframe, n=20):
        self.dataframe = dataframe
        self.n = n

    def get_top_n_items(self, user_items):
        item_frequencies = Counter(user_items)
        top_n_items = item_frequencies.most_common(self.n)
        return [item[0] for item in top_n_items]

    def get_top_n_recommendations(self):
        grouped_data = self.dataframe.groupby('customer_id')['good_foreign_id'].apply(list).reset_index()
        grouped_data['preds'] = grouped_data['good_foreign_id'].apply(self.get_top_n_items)
        return dict(zip(grouped_data['customer_id'], grouped_data['preds']))