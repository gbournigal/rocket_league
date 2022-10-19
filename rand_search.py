# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 15:13:36 2022

@author: gbournigal
"""

import gc  # garbage collection
from scipy.stats import uniform
from data_extract import data_load
from data_augmentation import mirror_board
from feature_engineer import distances, demolitions, cols_to_drop
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV


df = data_load(SEED=1)


df = distances(df)
df = demolitions(df)


### Hypertune ###


df = df.drop(columns=cols_to_drop)
gc.collect()


param_distribs = {
    "objective": ["binary"],
    "num_leaves": range(20, 200, 20),
    "n_estimators": range(100, 1000, 50),
    "learning_rate": uniform(0.01, 0.29),
    "max_depth": range(3, 12),
    "feature_fraction": uniform(0.2, 0.75),
    "subsample": [0.7],
    "subsample_freq": [8],
    "n_jobs": [-1],
    "reg_alpha": [1],
    "reg_lambda": [2],
    "min_child_samples": [90]    
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
    n_jobs=-1,
    return_train_score=True
)
rnd_search.fit(df.drop(columns=['team_A_scoring_within_10sec', 'team_B_scoring_within_10sec']).values, 
               df['team_A_scoring_within_10sec'].values)