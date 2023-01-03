# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 11:36:16 2022

@author: gbournigal
"""

import streamlit as st


st.set_page_config(
     page_title="""Rocket League Model Analysis""",
     page_icon="🚙",
     layout="wide",
 )

st.title("""🚙Rocket League Model Analysis""")

st.write("""
         The following app visualize and analize the model built for the Kaggle Competition
         [Tabular Playground Series - Oct 2022](https://discuss.streamlit.io/t/hyperlink-in-streamlit-without-markdown/7046/3).
         The challenge was to predict the probability of each team scoring within the next 10 seconds of the game given a snapshot from a Rocket League match.
         More information and the data used is available on the link. 
         
         The code for the data exploration and training is available in [this Github repository](https://github.com/gbournigal/rocket_league).
         
         This app will focus on the model selected and to see what probabilities it predicts to different games.
         """)