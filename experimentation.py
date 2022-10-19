# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 14:58:47 2022

@author: gbournigal
"""

import pickle
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate
from data_extract import data_load
from data_augmentation import mirror_board
from feature_engineer import distances, demolitions, cols_to_drop
import gc  # garbage collection


def cross_val_model(df,
                    model, 
                    model_name, 
                    params, 
                    SAMPLE=0.2, 
                    comment=""):
    
    cv_results = cross_validate(model, 
                                X=df.drop(columns=['team_A_scoring_within_10sec', 'team_B_scoring_within_10sec']).values,
                                y=df['team_B_scoring_within_10sec'].astype('int').values, 
                                cv=5,
                                n_jobs=5,
                                scoring="neg_log_loss",
                                return_train_score=True
                                )
    
    model_results = {
        'model': model_name,
        'params': params,
        'comment': comment,
        'SAMPLE': SAMPLE,
        'cv_results': cv_results
        }
    
    pickle.dump(model_results, open(f'results/model_{model_name}_results.pickle', 'wb'))
    


if __name__ == '__main__':
    SAMPLE=0.2

    df = data_load(SAMPLE=SAMPLE)

    # df = mirror_board(df)

    df = distances(df)
    df = demolitions(df)
    df = df.drop(columns=cols_to_drop)
    gc.collect()
    
    params = {
        'objective': 'binary',
        'num_leaves': 140, # was 128
        'n_estimators': 1000,
        'max_depth': 10, # was 10
        'learning_rate': 0.014045956, # was 0.1
        'feature_fraction': 0.7083506685757333, # was 0.75
        'subsample': 0.7,
        'subsample_freq': 8,
        'n_jobs': 5,
        'reg_alpha': 1,
        'reg_lambda': 2,
        'min_child_samples': 90,
    }
    model_b = LGBMClassifier(**params)
    cross_val_model(df,
                    model_b,
                    'LightGBM_hyper_20',
                    params,
                    SAMPLE,
                    )
    
    
    params_xgb = {
        'objective': 'binary:logistic',
        'tree_method': 'gpu_hist',
        'n_estimators': 500,
        'max_leaves': 140,
        'learning_rate': 0.2,
        'max_depth': 10,
        'subsample': 0.7
        }
    model_b = XGBClassifier(**params_xgb)
    cross_val_model(df,
                    model_b,
                    'XGBC_firsttry',
                    params_xgb,
                    SAMPLE,
                    )
    
    lightgbm_result = pickle.load(open('results/model_lightgbm_results.pickle', 'rb'))
    xgboost_result = pickle.load(open('results/model_XGBC_firsttry_results.pickle', 'rb'))
    

