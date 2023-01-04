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
         This app is designed to visualize and analyze the model built for the Kaggle Competition
         [Tabular Playground Series - Oct 2022](https://discuss.streamlit.io/t/hyperlink-in-streamlit-without-markdown/7046/3).
         The challenge was to predict the probability of each team scoring within the next 10 seconds of a Rocket League match, based on a snapshot of the game. You can find more information about the competition and the data used on the provided link. 
         
         The code for the data exploration and model training can be found in the accompanying [Github repository](https://github.com/gbournigal/rocket_league).
         
         The focus of this app is on the selected model, and how it predicts the probability of scoring in various games.
         
         To see the results, open the sidebar and navigate to the Model Analysis in the top left corner.
         """)