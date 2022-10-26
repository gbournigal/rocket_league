# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 14:58:47 2022

@author: gbournigal
"""

import pickle
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import log_loss
from data_extract import data_load
from data_augmentation import mirror_board
from feature_engineer import distances, demolitions, cols_to_drop, calc_speeds
import gc  # garbage collection


def evaluation_model(df,
                    df_val,
                    model, 
                    model_name, 
                    params, 
                    SAMPLE=0.2, 
                    comment=""):
    
    model.fit(
        X=df.drop(columns=['team_A_scoring_within_10sec', 'team_B_scoring_within_10sec']).values,
        y=df['team_B_scoring_within_10sec'].astype('int').values, 
        )
    
    val_pred = model.predict_proba(
        X=df_val.drop(columns=['team_A_scoring_within_10sec', 'team_B_scoring_within_10sec']).values
        )
    
    train_pred = model.predict_proba(
        X=df.drop(columns=['team_A_scoring_within_10sec', 'team_B_scoring_within_10sec']).values
        )
    
    train_score = log_loss(df['team_B_scoring_within_10sec'].astype('int').values,
                         train_pred)
    
    val_score = log_loss(df_val['team_B_scoring_within_10sec'].astype('int').values,
                         val_pred)

    
    model_results = {
        'model_name': model_name,
        'params': params,
        'comment': comment,
        'SAMPLE': SAMPLE,
        'train_score': train_score,
        'val_score': val_score,
        'model': model
        }
    
    pickle.dump(model_results, open(f'results/model_{model_name}_results.pickle', 'wb'))
    

if __name__ == '__main__':
    SAMPLE=0.5

    df = data_load(SAMPLE=SAMPLE, df_size='experimentation')
    
    df_eval = data_load(SAMPLE=1, df_size='validation')
    df_eval_simple = df_eval.copy()
    
    dfs = {'df': df,
           'df_eval_simple': df_eval_simple}

    # df = mirror_board(df)
    for i in dfs.keys():
        dfs[i] = distances(dfs[i])
        dfs[i] = demolitions(dfs[i])
        dfs[i] = calc_speeds(dfs[i])
        dfs[i] = dfs[i].drop(columns=cols_to_drop)
    gc.collect()
    
    params = {
        'objective': 'binary',
        'num_leaves': 140, # was 128
        'n_estimators': 1500, # was 1000
        'max_depth': 7, # was 10
        'learning_rate': 0.03, # was 0.1
        'feature_fraction': 0.664079, # was 0.75
        'subsample': 0.7,
        'subsample_freq': 8,
        'n_jobs': -1,
        'reg_alpha': 0, # was 1
        'reg_lambda': 2, # was 2
        'min_child_samples': 50,
    }
    model_b = LGBMClassifier(**params)
    evaluation_model(dfs['df'],
                     dfs['df_eval_simple'],
                     model_b,
                     'LightGBM_hyper_50_new_train',
                     params,
                     SAMPLE,
                    )
    
    
    params_xgb = {
        'objective': 'binary:logistic',
        'tree_method': 'gpu_hist',
        'n_estimators': 1800,
        'colsample_bytree': 0.562821,
        'learning_rate': 0.0130056,
        'max_depth': 7,
        'alpha': 1.5,
        'lambda': 1.5,
        'gamma': 0.2
        }
    model_b = XGBClassifier(**params_xgb)
    evaluation_model(dfs['df'],
                     dfs['df_eval_simple'],
                     model_b,
                     'XGBC_50_hyper_speed',
                     params_xgb,
                     SAMPLE,
                    )
    
    
    
    lightgbm_result = pickle.load(open('results/model_LightGBM_hyper_50_eval_results.pickle', 'rb'))
    lightgbm_hyper_result_new = pickle.load(open('results/model_LightGBM_hyper_50_new_train_results.pickle', 'rb'))
    xgboost_hyper_3_result = pickle.load(open('results/model_XGBC_50_hyper_results.pickle', 'rb'))
    xgboost_hyper_3_result_nf = pickle.load(open('results/model_XGBC_50_hyper_speed_results.pickle', 'rb'))
    
    rnd_search = pickle.load(open('results/rnd_search_lgbm_results.pickle', 'rb'))
    
    

