# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 15:13:36 2022

@author: gbournigal
"""

import pickle
import pandas as pd
import gc  # garbage collection
from scipy.stats import uniform
from data_extract import data_load
from data_augmentation import mirror_board
from feature_engineer import distances, demolitions, cols_to_drop
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV


df = data_load(SAMPLE=0.2, df_size='experimentation')


df = distances(df)
df = demolitions(df)


### Hypertune ###


df = df.drop(columns=cols_to_drop)
gc.collect()


param_distribs = {
    "objective": ["binary"],
    "num_leaves": [140],
    "n_estimators": range(500, 2000, 100),
    "learning_rate": uniform(0.01, 0.2),
    "max_depth": range(3, 12),
    "feature_fraction": uniform(0.5, 0.25),
    "subsample": [0.7],
    "subsample_freq": [8],
    "n_jobs": [4],
    "reg_alpha": [0, 1, 2],
    'lambda': [0, 1, 2],
    "min_child_samples": range(10, 120, 10),  
}


lgbm = LGBMClassifier()
rnd_search = RandomizedSearchCV(
    lgbm,
    param_distributions=param_distribs,
    n_iter=100,
    cv=5,
    scoring="neg_log_loss",
    verbose=2,
    random_state=1,
    n_jobs=3,
    return_train_score=True
)
rnd_search.fit(df.drop(columns=['team_A_scoring_within_10sec', 'team_B_scoring_within_10sec']).values, 
               df['team_A_scoring_within_10sec'].values)


param_distribs = {
    "max_depth": range(3, 11),
    "subsample": uniform(0.5, 0.3),
    "n_estimators": range(500, 2000, 100),
    "learning_rate": uniform(0.01, 0.29),
    'gamma': [0, 0.2, 0.4],
    "colsample_bytree": uniform(0.5, 0.5),
    'lambda': [0, 1, 1.5],
    "alpha": [0, 1, 1.5],
    'objective': ['binary:logistic']
}


xgb = XGBClassifier()
rnd_search = RandomizedSearchCV(
    xgb,
    param_distributions=param_distribs,
    n_iter=100,
    cv=5,
    scoring="neg_log_loss",
    verbose=2,
    random_state=1,
    n_jobs=1,
    return_train_score=True
)
rnd_search.fit(df.drop(columns=['team_A_scoring_within_10sec', 'team_B_scoring_within_10sec']).values, 
               df['team_A_scoring_within_10sec'].astype('int').values)


results_rnd = pd.DataFrame(rnd_search.cv_results_)
pickle.dump(results_rnd, open('results/rnd_search_xgb1_results.pickle', 'wb'))

