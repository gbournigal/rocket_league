# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 10:19:04 2022

@author: gbournigal
"""

import pickle
import gc
from data_extract import data_load
from feature_engineer import (distances, 
                              demolitions,
                              cols_to_drop,
                              calc_speeds,
                              min_dist_to_goal,
                              max_dist_to_goal,
                              mean_dist_to_goal,
                              add_angle_features)
from data_augmentation import mirror_board, mirror_x
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def final_models(df,
                model_a,
                model_b):
    
    target_A = df['team_A_scoring_within_10sec'].astype('int').values
    target_B = df['team_B_scoring_within_10sec'].astype('int').values
    
    del df['team_A_scoring_within_10sec']
    del df['team_B_scoring_within_10sec']
    
    df = df.to_numpy()
    
    model_b.fit(
        X=df,
        y=target_B, 
        )
    
    model_a.fit(
        X=df,
        y=target_A, 
        )
    model_b.booster_.save_model("model_b.json")
    model_a.booster_.save_model("model_a.json")
    
if __name__ == '__main__':
    SAMPLE=1

    df = data_load(SAMPLE=SAMPLE, df_size='full')
    # df = mirror_board(df, percentage=0.75)
    # df = mirror_x(df, percentage=0.15)
    
    df = distances(df)
    df = demolitions(df)
    df = calc_speeds(df)
    df = min_dist_to_goal(df)
    df = add_angle_features(df)
    df = max_dist_to_goal(df)
    df = mean_dist_to_goal(df)
    df = df.drop(columns=cols_to_drop)
    gc.collect()
    
    # params_xgb = {
    #     'objective': 'binary:logistic',
    #     'tree_method': 'gpu_hist',
    #     'n_estimators': 1800,
    #     'colsample_bytree': 0.562821,
    #     'learning_rate': 0.0130056,
    #     'max_depth': 7,
    #     'alpha': 1.5,
    #     'lambda': 1.5,
    #     'gamma': 0.2
    #     }
    
    # model_a = XGBClassifier(**params_xgb)
    # model_b = XGBClassifier(**params_xgb)
    
    params_lgbm = {
        'objective': 'binary',
        'num_leaves': 140, # was 128
        'n_estimators': 1500, # was 1000
        'max_depth': 7, # was 10
        'learning_rate': 0.03, # was 0.1
        'feature_fraction': 0.664079, # was 0.75
        'subsample': 0.7,
        'subsample_freq': 8,
        'n_jobs': 8,
        'reg_alpha': 0, # was 1
        'reg_lambda': 2, # was 2
        'min_child_samples': 50,
    }
    
    model_a = LGBMClassifier(**params_lgbm)
    model_b = LGBMClassifier(**params_lgbm)
    
    final_models(df,
                 model_a,
                 model_b)