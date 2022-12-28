# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 15:44:51 2022

@author: gbournigal
"""

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


train = dt.fread('C:/Users/gbournigal/Documents/GitHub/rocket_league/data/train_0.csv').to_pandas()
