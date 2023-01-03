# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 15:44:51 2022

@author: gbournigal
"""

import pickle
import streamlit as st
import warnings 
warnings.filterwarnings('ignore')

import datatable as dt
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.patches as patches
from matplotlib.patches import Wedge

import imageio

from glob import glob
from matplotlib import animation, rc
from feature_engineer import distances, demolitions, calc_speeds, min_dist_to_goal, add_angle_features, mean_dist_to_goal, max_dist_to_goal, cols_to_drop



@st.cache
def read_sample_data():
    return dt.fread('data/train_sample.csv').to_pandas()


train = read_sample_data()
# train = train.sample(10000)
# train.to_csv('data/train_sample.csv')
dtypes_dict_train = dict(pd.read_csv('data/train_dtypes.csv').values)

# Reduce memory usage by 70%.
train = train.astype(dtypes_dict_train)

def draw_rocket_league_field() -> (matplotlib.figure.Figure, matplotlib.axes.SubplotBase):
    """
    Draws the rocket league playfield for Kaggle TPS Oct., 2022.
    :return matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot:
    """
    # Draws two field rectangles ('Player Field Border', 'Ball Field Border').
    center = (0, 0)
    vertices = []
    codes = []

    codes_pl = [Path.MOVETO] + [Path.LINETO]*3 + [Path.CLOSEPOLY]
    vertices_pl = [(-120, -82.5), (120, -82.5),
                   (120, 82.5), (-120, 82.5), center] # inversed x, y.

    codes_ball = [Path.MOVETO] + [Path.LINETO]*3 + [Path.CLOSEPOLY]
    vertices_ball = [(-104.3125, -80.8125), (104.3125, -80.8125),
                     (104.3125, 80.6875), (-104.3125, 80.6875), center] # inversed x, y. 

    path_pl = Path(vertices_pl, codes_pl)
    path_ball = Path(vertices_ball, codes_ball)

    pathpatch_pl = PathPatch(path_pl, facecolor='#676C40', edgecolor='#516EA7', 
                             linewidth=2, label='Player Field Border')
    pathpatch_ball = PathPatch(path_ball, facecolor='none', edgecolor='yellow', 
                               linewidth=1, label='Ball Field Border')


    # Draws the center outer and inner circles.
    center_outer_circle = patches.CirclePolygon(center, 20, color='white',
                                                fill=False, alpha=0.7)
    theta1, theta2 = 90, 90 + 180
    center_half_inner_circle_blue = Wedge(center, 10, theta1,  theta2, fc='#516EA7', alpha=0.6)
    center_half_inner_circle_orange = Wedge(center, 10, theta2,  theta1, fc='#C57E31', alpha=0.6)

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.add_patch(pathpatch_pl)

    # Draws the dark green stripes.
    for i in range(6):
        dgreen_stripe = patches.Rectangle((-110 + i*40, -80), 20, 160, color='#565B39')
        ax.add_patch(dgreen_stripe)

    # Draws the white corner triangles.
    corners = [[1, 1], [-1, 1], [1, -1], [-1, -1]]
    for corner in corners:
        x, y = corner[0], corner[1]
        corner_triangle = patches.Polygon([(121*x, 85*y), (105*x, 85*y), (121*x, 55*y)], 40, color='white')
        ax.add_patch(corner_triangle)

    # Draws the blue and orange outer goalie boxes.
    for i, color in enumerate(['#617A95', '#8D5C2C']):
        goal_outer_box = patches.Rectangle((-120 + i*200, -50), 40, 100, color=color, alpha=0.7, ec='white')
        ax.add_patch(goal_outer_box)

    # Draws the blue and orange inner goalie boxes.
    for i in range(2):
        goal_inner_box = patches.Rectangle((-120 + i*230, -25), 10, 50, color='none', ec='white')
        ax.add_patch(goal_inner_box)

    # Draws center white cross.
    plt.plot([-70, 70], [0, 0], color='white', alpha=0.7)
    plt.plot([0, 0], [-80.5, 80.5], color='white', alpha=0.7)

    # Adds the patches from above drawings.
    ax.add_patch(pathpatch_ball)
    ax.add_patch(center_outer_circle)
    ax.add_patch(center_half_inner_circle_blue)
    ax.add_patch(center_half_inner_circle_orange)

    # Adds the title, legend and removes x, y-axis with the respective ticks.
    for pos in ['top', 'bottom', 'left', 'right']:
        ax.spines[pos].set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title('Playfield')
    plt.legend(handles=[pathpatch_pl, pathpatch_ball], facecolor='#676C40',
               labelcolor='white', loc='upper right', bbox_to_anchor=(0.964, 1.05))
    return fig, ax


def main(game_num: int, event_id: int, event_time: str) -> None:
    """
    Draws the starting positions or the consequtive frames of
    the specified game_num, event_id and event_time (optional).
    :param game_num: int
    :param event_id: int
    :param event_time: str
    :return: None
    """
    fig, ax = draw_rocket_league_field()
    player_positions_y = train.columns[train.columns.str.contains('(^[p0-9_]+)([pos_]+x)')]
    player_positions_x = train.columns[train.columns.str.contains('(^[p0-9_]+)([pos_]+y)')]
    ball_positions_y = 'ball_pos_x'
    ball_positions_x = 'ball_pos_y'
    
    if event_time:
        game = train.query(f'game_num == {game_num} and event_id == {event_id} and event_time == {event_time}')
        title = f'game #{game_num} event_id {event_id} event_time {event_time:.2f}'
    else:
        game = train.query(f'game_num == {game_num} and event_id == {event_id}')
        title = f'game #{game_num} event_id {event_id}'
        
    x_coordinates_ball = pd.melt(game, id_vars=['game_num', 'event_time'], value_vars=ball_positions_x, 
                            var_name='ball_pos_x', value_name='X')
    y_coordinates_ball = pd.melt(game, id_vars=['game_num', 'event_time'], value_vars=ball_positions_y, 
                            var_name='ball_pos_y', value_name='Y')
    x_coordinates_pl = pd.melt(game, id_vars=['game_num', 'event_time'], value_vars=player_positions_x, 
                            var_name='player_pos_x', value_name='X')
    y_coordinates_pl = pd.melt(game, id_vars=['game_num', 'event_time'], value_vars=player_positions_y, 
                            var_name='player_pos_y', value_name='Y')

    game_ball = pd.concat([x_coordinates_ball, y_coordinates_ball['Y']], axis=1)[['game_num', 'event_time', 'X', 'Y']]
    game_pl = pd.concat([x_coordinates_pl, y_coordinates_pl['Y']], axis=1)[['game_num', 'event_time', 'player_pos_x', 'X', 'Y']]
    game_pl['player'] = game_pl.player_pos_x.str.extract('([0-9])+') 
    sns.scatterplot(data=game_pl.iloc[:int(game_pl.shape[0]/2), :], x='X', y='Y', ax=ax, marker='o', s=45, hue='player', palette='Blues_r')
    sns.scatterplot(data=game_pl.iloc[-int(game_pl.shape[0]/2):, :], x='X', y='Y', ax=ax, marker='o', s=45, hue='player', palette='Oranges_r')

    sns.scatterplot(data=game_ball, x='X', y='Y', ax=ax, color='red', marker='*', s=150, label='Ball')
    ax.legend(facecolor='#676C40', labelcolor='white', loc='upper right', bbox_to_anchor=(1.2, 1.04))
    plt.title(title)
    return fig, ax
    


first_row = train.head(1)
fig, ax = main(first_row['game_num'][0], first_row['event_id'][0], first_row['event_time'][0])

first_row = distances(first_row)
first_row = demolitions(first_row)
first_row = calc_speeds(first_row)
first_row = min_dist_to_goal(first_row)
first_row = add_angle_features(first_row)
first_row = max_dist_to_goal(first_row)
first_row = mean_dist_to_goal(first_row)
first_row = first_row.drop(columns=cols_to_drop)

models = pickle.load(open('results/final_models.pickle', 'rb'))

model_a = models['model_a']
model_b = models['model_b']
st.title("""🤖Model Analysis""")
st.pyplot(fig)


