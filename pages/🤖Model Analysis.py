# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 15:44:51 2022

@author: gbournigal
"""


import warnings

warnings.filterwarnings("ignore")
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import streamlit as st
import plotly.express as px
import datatable as dt
import pandas as pd
import numpy as np
import lightgbm as lgb
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.patches import Wedge
from feature_engineer import (
    distances,
    demolitions,
    calc_speeds,
    min_dist_to_goal,
    add_angle_features,
    mean_dist_to_goal,
    max_dist_to_goal,
    cols_to_drop,
)

cols_to_drop_full = cols_to_drop + [
    "C0",
    "team_B_scoring_within_10sec",
    "team_A_scoring_within_10sec",
]


@st.cache
def read_sample_data():
    return dt.fread("data/train_sample.csv").to_pandas()


def draw_rocket_league_field() -> (
    matplotlib.figure.Figure,
    matplotlib.axes.SubplotBase,
):
    """
    Draws the rocket league playfield for Kaggle TPS Oct., 2022.
    :return matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot:
    """
    # Draws two field rectangles ('Player Field Border', 'Ball Field Border').
    center = (0, 0)
    vertices = []
    codes = []

    codes_pl = [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
    vertices_pl = [
        (-120, -82.5),
        (120, -82.5),
        (120, 82.5),
        (-120, 82.5),
        center,
    ]  # inversed x, y.

    codes_ball = [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
    vertices_ball = [
        (-104.3125, -80.8125),
        (104.3125, -80.8125),
        (104.3125, 80.6875),
        (-104.3125, 80.6875),
        center,
    ]  # inversed x, y.

    path_pl = Path(vertices_pl, codes_pl)
    path_ball = Path(vertices_ball, codes_ball)

    pathpatch_pl = PathPatch(
        path_pl,
        facecolor="#676C40",
        edgecolor="#516EA7",
        linewidth=2,
        label="Player Field Border",
    )
    pathpatch_ball = PathPatch(
        path_ball,
        facecolor="none",
        edgecolor="yellow",
        linewidth=1,
        label="Ball Field Border",
    )

    # Draws the center outer and inner circles.
    center_outer_circle = patches.CirclePolygon(
        center, 20, color="white", fill=False, alpha=0.7
    )
    theta1, theta2 = 90, 90 + 180
    center_half_inner_circle_blue = Wedge(
        center, 10, theta1, theta2, fc="#516EA7", alpha=0.6
    )
    center_half_inner_circle_orange = Wedge(
        center, 10, theta2, theta1, fc="#C57E31", alpha=0.6
    )

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.add_patch(pathpatch_pl)

    # Draws the dark green stripes.
    for i in range(6):
        dgreen_stripe = patches.Rectangle(
            (-110 + i * 40, -80), 20, 160, color="#565B39"
        )
        ax.add_patch(dgreen_stripe)

    # Draws the white corner triangles.
    corners = [[1, 1], [-1, 1], [1, -1], [-1, -1]]
    for corner in corners:
        x, y = corner[0], corner[1]
        corner_triangle = patches.Polygon(
            [(121 * x, 85 * y), (105 * x, 85 * y), (121 * x, 55 * y)],
            40,
            color="white",
        )
        ax.add_patch(corner_triangle)

    # Draws the blue and orange outer goalie boxes.
    for i, color in enumerate(["#617A95", "#8D5C2C"]):
        goal_outer_box = patches.Rectangle(
            (-120 + i * 200, -50), 40, 100, color=color, alpha=0.7, ec="white"
        )
        ax.add_patch(goal_outer_box)

    # Draws the blue and orange inner goalie boxes.
    for i in range(2):
        goal_inner_box = patches.Rectangle(
            (-120 + i * 230, -25), 10, 50, color="none", ec="white"
        )
        ax.add_patch(goal_inner_box)

    # Draws center white cross.
    plt.plot([-70, 70], [0, 0], color="white", alpha=0.7)
    plt.plot([0, 0], [-80.5, 80.5], color="white", alpha=0.7)

    # Adds the patches from above drawings.
    ax.add_patch(pathpatch_ball)
    ax.add_patch(center_outer_circle)
    ax.add_patch(center_half_inner_circle_blue)
    ax.add_patch(center_half_inner_circle_orange)

    # Adds the title, legend and removes x, y-axis with the respective ticks.
    for pos in ["top", "bottom", "left", "right"]:
        ax.spines[pos].set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title("Playfield")
    plt.legend(
        handles=[pathpatch_pl, pathpatch_ball],
        facecolor="#676C40",
        labelcolor="white",
        loc="upper right",
        bbox_to_anchor=(0.964, 1.05),
    )
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
    player_positions_y = train.columns[
        train.columns.str.contains("(^[p0-9_]+)([pos_]+x)")
    ]
    player_positions_x = train.columns[
        train.columns.str.contains("(^[p0-9_]+)([pos_]+y)")
    ]
    ball_positions_y = "ball_pos_x"
    ball_positions_x = "ball_pos_y"

    if event_time:
        game = train.query(
            f"game_num == {game_num} and event_id == {event_id} and event_time == {event_time}"
        )
        title = (
            f"game #{game_num} event_id {event_id} event_time {event_time:.2f}"
        )
    else:
        game = train.query(
            f"game_num == {game_num} and event_id == {event_id}"
        )
        title = f"game #{game_num} event_id {event_id}"

    x_coordinates_ball = pd.melt(
        game,
        id_vars=["game_num", "event_time"],
        value_vars=ball_positions_x,
        var_name="ball_pos_x",
        value_name="X",
    )
    y_coordinates_ball = pd.melt(
        game,
        id_vars=["game_num", "event_time"],
        value_vars=ball_positions_y,
        var_name="ball_pos_y",
        value_name="Y",
    )
    x_coordinates_pl = pd.melt(
        game,
        id_vars=["game_num", "event_time"],
        value_vars=player_positions_x,
        var_name="player_pos_x",
        value_name="X",
    )
    y_coordinates_pl = pd.melt(
        game,
        id_vars=["game_num", "event_time"],
        value_vars=player_positions_y,
        var_name="player_pos_y",
        value_name="Y",
    )

    game_ball = pd.concat(
        [x_coordinates_ball, y_coordinates_ball["Y"]], axis=1
    )[["game_num", "event_time", "X", "Y"]]
    game_pl = pd.concat([x_coordinates_pl, y_coordinates_pl["Y"]], axis=1)[
        ["game_num", "event_time", "player_pos_x", "X", "Y"]
    ]
    game_pl["player"] = game_pl.player_pos_x.str.extract("([0-9])+")
    sns.scatterplot(
        data=game_pl.iloc[: int(game_pl.shape[0] / 2), :],
        x="X",
        y="Y",
        ax=ax,
        marker="o",
        s=45,
        hue="player",
        palette="Blues_r",
    )
    sns.scatterplot(
        data=game_pl.iloc[-int(game_pl.shape[0] / 2) :, :],
        x="X",
        y="Y",
        ax=ax,
        marker="o",
        s=45,
        hue="player",
        palette="Oranges_r",
    )

    sns.scatterplot(
        data=game_ball,
        x="X",
        y="Y",
        ax=ax,
        color="red",
        marker="*",
        s=150,
        label="Ball",
    )
    ax.legend(
        facecolor="#676C40",
        labelcolor="white",
        loc="upper right",
        bbox_to_anchor=(1.2, 1.04),
    )
    plt.title(title)
    return fig, ax


@st.cache
def data_with_probs(train):
    player_positions_y = train.columns[
        train.columns.str.contains("(^[p0-9_]+)([pos_]+x)")
    ]
    player_positions_x = train.columns[
        train.columns.str.contains("(^[p0-9_]+)([pos_]+y)")
    ]
    ball_positions_y = "ball_pos_x"
    ball_positions_x = "ball_pos_y"
    viz = train.copy()

    train = distances(train)
    train = demolitions(train)
    train = calc_speeds(train)
    train = min_dist_to_goal(train)
    train = add_angle_features(train)
    train = max_dist_to_goal(train)
    train = mean_dist_to_goal(train)
    train = train.drop(columns=cols_to_drop_full)

    model_a = lgb.Booster(model_file="model_a.json")
    model_b = lgb.Booster(model_file="model_b.json")

    viz["prob_a"] = model_a.predict(train)
    viz["prob_b"] = model_b.predict(train)

    viz = viz[
        list(player_positions_x)
        + list(player_positions_y)
        + list([ball_positions_x])
        + list([ball_positions_y])
        + [
            "prob_a",
            "prob_b",
            "team_B_scoring_within_10sec",
            "team_A_scoring_within_10sec",
            "game_num",
            "event_id",
            "event_time",
        ]
    ]
    return viz


def plot_probs(game_snap):
    plot_probs = game_snap[["prob_a", "prob_b"]].transpose()
    plot_probs.rename(
        columns={plot_probs.columns[0]: "probabilities"}, inplace=True
    )
    plot_probs.reset_index(inplace=True)
    plot_probs["index"] = np.where(
        plot_probs["index"] == "prob_a", "Team A", "Team_B"
    )
    fig = px.bar(
        plot_probs,
        x="index",
        y="probabilities",
        title="Goal Probabilities",
        labels={"probabilities": "Probabilities of Scoring", "index": "Team"},
    )
    fig.update_layout(yaxis_tickformat=".2%")
    st.plotly_chart(fig, use_container_width=True)


def result_text(game_snap):
    message = np.where(
        game_snap["team_A_scoring_within_10sec"] == 1,
        "Actual result: Team A scored within 10 seconds",
        np.where(
            game_snap["team_B_scoring_within_10sec"] == 1,
            "Actual result: Team B scored within 10 seconds",
            "Actual result: No goals in the next 10 seconds",
        ),
    )
    prediction = np.where(
        game_snap["prob_a"] > model_threshold / 100,
        f'The model predicted a goal by team A with probability of {float(game_snap["prob_a"]):.2%}',
        np.where(
            game_snap["prob_b"] > model_threshold / 100,
            f'The model predicted a goal by team B with probability of {float(game_snap["prob_b"]):.2%}',
            f"The model predicted no goals, based on the selected threshold of {model_threshold}%",
        ),
    )
    st.text(prediction[0])
    st.text(message[0])


def define_page(df):
    if len(df) == 0:
        st.write("No data found with the selected filters")
    else:
        game_snap = df.sample(1)
        fig, ax = main(
            int(game_snap["game_num"]),
            int(game_snap["event_id"]),
            float(game_snap["event_time"]),
        )
        st.pyplot(fig)
        plot_probs(game_snap)
        result_text(game_snap)


st.title("""🤖Model Analysis""")
train = read_sample_data()
if st.button("Next Image"):
    train = read_sample_data()

dtypes_dict_train = dict(pd.read_csv("data/train_dtypes.csv").values)

# Reduce memory usage by 70%.
train = train.astype(dtypes_dict_train)
viz = data_with_probs(train)
st.sidebar.write("Data Filters")
resultado = st.sidebar.radio(
    "Actual Result",
    ["Any", "No Goals", "Team A Goal", "Team B Goal"],
)

if resultado == "Team A Goal":
    viz = viz[viz["team_A_scoring_within_10sec"] == 1]
elif resultado == "Team B Goal":
    viz = viz[viz["team_B_scoring_within_10sec"] == 1]
elif resultado == "No Goals":
    viz = viz[
        (viz["team_A_scoring_within_10sec"] == 0)
        & (viz["team_B_scoring_within_10sec"] == 0)
    ]

model_result = st.sidebar.radio(
    "Model Result (result of prediction based on threshold):",
    ["Any", "Right", "Wrong"],
)

model_threshold = st.sidebar.slider(
    "Model Threshold (% needed to make prediction):", 1, 99, 25
)

if model_result == "Right":
    if resultado == "Team A Goal":
        viz = viz[viz["prob_a"] > model_threshold / 100]
    elif resultado == "Team B Goal":
        viz = viz[viz["prob_b"] > model_threshold / 100]
    elif resultado == "No Goals":
        viz = viz[
            (viz["prob_a"] < model_threshold / 100)
            & (viz["prob_b"] < model_threshold / 100)
        ]
    elif resultado == "Any":
        viz = viz[
            (
                (viz["prob_a"] > model_threshold / 100)
                & (viz["team_A_scoring_within_10sec"] == 1)
            )
            | (
                (viz["prob_b"] > model_threshold / 100)
                & (viz["team_B_scoring_within_10sec"] == 1)
            )
            | (
                (
                    (viz["prob_a"] < model_threshold / 100)
                    & (viz["prob_b"] < model_threshold / 100)
                )
                & (
                    (viz["team_B_scoring_within_10sec"] == 0)
                    & (viz["team_A_scoring_within_10sec"] == 0)
                )
            )
        ]

elif model_result == "Wrong":
    if resultado == "Team A Goal":
        viz = viz[viz["prob_a"] < model_threshold / 100]
    elif resultado == "Team B Goal":
        viz = viz[viz["prob_b"] < model_threshold / 100]
    elif resultado == "No Goals":
        viz = viz[
            (viz["prob_a"] > model_threshold / 100)
            | (viz["prob_b"] > model_threshold / 100)
        ]
    elif resultado == "Any":
        viz = viz[
            (
                (viz["prob_a"] < model_threshold / 100)
                & (viz["team_A_scoring_within_10sec"] == 1)
            )
            | (
                (viz["prob_b"] < model_threshold / 100)
                & (viz["team_B_scoring_within_10sec"] == 1)
            )
            | (
                (
                    (viz["prob_a"] > model_threshold / 100)
                    & (viz["prob_b"] > model_threshold / 100)
                )
                & (
                    (viz["team_B_scoring_within_10sec"] == 0)
                    & (viz["team_A_scoring_within_10sec"] == 0)
                )
            )
        ]


define_page(viz)
