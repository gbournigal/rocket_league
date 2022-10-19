# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 15:18:43 2022

@author: gbournigal
"""


import pandas as pd
import numpy as np
from feature_engineer import distances, demolitions

### TEST ###

model_a = 'PENDIENTE DEFINIRLO'
model_b = 'PENDIENTE DEFINIRLO'

df_test = pd.read_csv('data/test.csv')
df_test = distances(df_test)
df_test = demolitions(df_test)
df_test = df_test.drop(columns=['ball_pos_ball_dist',
                                'goal_A_pos_x',
                                'goal_B_pos_x',
                                'goal_A_pos_y',
                                'goal_B_pos_y',
                                'goal_A_pos_z',
                                'goal_B_pos_z'])



from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')

pred_a = np.zeros(df_test.shape[0])
pred_a_X = imp.fit_transform(df_test.drop(columns=['id']))
pred_a += model_a.predict_proba(pred_a_X)[:, 1]


pred_b = np.zeros(df_test.shape[0])
pred_b_X = imp.fit_transform(df_test.drop(columns=['id']))
pred_b += model_b.predict_proba(pred_b_X)[:, 1]



df_submission = pd.DataFrame(
    {
        "id": df_test['id'],
        "team_A_scoring_within_10sec": pred_a,
        "team_B_scoring_within_10sec": pred_b
    }
)

df_submission.to_csv('submission.csv', index=False)