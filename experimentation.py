# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 14:58:47 2022

@author: gbournigal
"""

from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_validate
from data_extract import data_load
from data_augmentation import mirror_board
from feature_engineer import distances, demolitions, cols_to_drop
import gc  # garbage collection

df = data_load(SAMPLE=0.2)

# df = mirror_board(df)




df = distances(df)
df = demolitions(df)
df = df.drop(columns=cols_to_drop)
gc.collect()


# LightGBM
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


cv_results = cross_validate(model_b, 
                            X=df.drop(columns=['team_A_scoring_within_10sec', 'team_B_scoring_within_10sec']).values,
                            y=df['team_B_scoring_within_10sec'].values, 
                            cv=5,
                            n_jobs=5,
                            scoring="neg_log_loss",
                            return_train_score=True
                            )




