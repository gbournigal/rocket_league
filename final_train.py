# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 10:19:04 2022

@author: gbournigal
"""

import pickle
import gc
from data_extract import data_load
from feature_engineer import distances, demolitions, cols_to_drop
from xgboost import XGBClassifier


def final_models(df,
                model_a,
                model_b):
    
    model_b.fit(
        X=df.drop(columns=['team_A_scoring_within_10sec', 'team_B_scoring_within_10sec']).values,
        y=df['team_B_scoring_within_10sec'].astype('int').values, 
        )
    
    model_a.fit(
        X=df.drop(columns=['team_A_scoring_within_10sec', 'team_B_scoring_within_10sec']).values,
        y=df['team_A_scoring_within_10sec'].astype('int').values, 
        )
    pickle.dump({'model_a': model_a,
                 'model_b': model_b}, open('results/final_models.pickle', 'wb'))
    
    
if __name__ == '__main__':
    SAMPLE=1

    df = data_load(SAMPLE=SAMPLE, df_size='full')
    
    df = distances(df)
    df = demolitions(df)
    df = df.drop(columns=cols_to_drop)
    gc.collect()
    
    params_xgb = {
        'objective': 'binary:logistic',
        'tree_method': 'gpu_hist',
        'n_estimators': 800,
        'colsample_bytree': 0.848755,
        'learning_rate': 0.0205762,
        'max_depth': 9,
        'alpha': 1,
        'lambda': 0,
        'gamma': 0
        }
    
    model_a = XGBClassifier(**params_xgb)
    model_b = XGBClassifier(**params_xgb)
    final_models(df,
                 model_a,
                 model_b)