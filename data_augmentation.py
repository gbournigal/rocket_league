# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 14:56:08 2022

@author: gbournigal
"""

import pandas as pd

def mirror_ball(data, only_x=False):
    # Mirror the coordinates
    if only_x is False:
        data["ball_pos_y"] = data["ball_pos_y"] * -1
        data["ball_vel_y"] = data["ball_vel_y"] * -1
    data["ball_pos_x"] = data["ball_pos_x"] * -1
    data["ball_vel_x"] = data["ball_vel_x"] * -1
    return data
    
def mirror_players(data, only_x=False):
    # Mirror the coordinates
    def mirror(data, p, a, only_x=False):
        if only_x is False:
            possible_coordinates = ['x', 'y']
        else:
            possible_coordinates = ['x']
        
        if a in possible_coordinates:
            data[f"p{p}_pos_{a}"] = data[f"p{p}_pos_{a}"] * -1
            data[f"p{p+3}_pos_{a}"] = data[f"p{p+3}_pos_{a}"] * -1
            data[f"p{p}_vel_{a}"] = data[f"p{p}_vel_{a}"] * -1
            data[f"p{p+3}_vel_{a}"] = data[f"p{p+3}_vel_{a}"] * -1
            
            tmp= data[f"p{p}_pos_{a}"].copy()
            data[f"p{p}_pos_{a}"] = data[f"p{p+3}_pos_{a}"]
            data[f"p{p+3}_pos_{a}"] = tmp
    
            tmp= data[f"p{p}_vel_{a}"].copy()
            data[f"p{p}_vel_{a}"] = data[f"p{p+3}_vel_{a}"]
            data[f"p{p+3}_vel_{a}"] = tmp
            
        if a == 'z':
            tmp= data[f"p{p}_pos_z"].copy()
            data[f"p{p}_pos_z"] = data[f"p{p+3}_pos_z"]
            data[f"p{p+3}_pos_z"] = tmp
    
            tmp= data[f"p{p}_vel_z"].copy()
            data[f"p{p}_vel_z"] = data[f"p{p+3}_vel_z"]
            data[f"p{p+3}_vel_z"] = tmp            
        return data
    
    for p in range(3):
        data = mirror(data, p, "y")
        data = mirror(data, p, "x")
        data = mirror(data, p, "z")
    return data


def mirror_others(data):
    for p in range(3):
        tmp= data[f"boost{p}_timer"].copy()
        data[f"boost{p}_timer"] = data[f"boost{p+3}_timer"]
        data[f"boost{p+3}_timer"] = tmp
        
        tmp= data[f"p{p}_boost"].copy()
        data[f"p{p}_boost"] = data[f"p{p+3}_boost"]
        data[f"p{p+3}_boost"] = tmp
    
    
    tmp= data["team_A_scoring_within_10sec"].copy()
    data["team_A_scoring_within_10sec"] = data["team_B_scoring_within_10sec"]
    data["team_B_scoring_within_10sec"] = tmp
    return data


def mirror_board(data, percentage=1, random_seed=24):
    mirrordata = data.copy()
    mirrordata = mirror_ball(mirrordata)
    mirrordata = mirror_players(mirrordata)
    mirrordata = mirror_others(mirrordata)
    mirrordata = mirrordata.sample(frac=percentage, random_state=random_seed)
    data = pd.concat([data, mirrordata])
    data = data.reset_index(drop=True)
    return data


def mirror_x(data, percentage=1, random_seed=42):
    mirrordata = data.copy()
    mirrordata = mirror_ball(mirrordata, only_x=True)
    mirrordata = mirror_players(mirrordata, only_x=True)
    mirrordata = mirrordata.sample(frac=percentage, random_state=random_seed)
    data = pd.concat([data, mirrordata])
    data = data.reset_index(drop=True)
    return data
