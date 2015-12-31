# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 18:12:44 2015

@author: datak_000
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

train_data = pd.read_csv("C:\\Users\\datak_000\\Documents\\OneNote Notebooks\\MDS549 - Data Mining Project\\west_nile\\input\\train.csv",
                          parse_dates=True, infer_datetime_format=True)

f = lambda x: dt.datetime.strptime(x, "%Y-%m-%d").strftime("%Y-%m")
year_f = lambda x: dt.datetime.strptime(x, "%Y-%m-%d").strftime("%Y")
year_w = lambda x: dt.datetime.strptime(x, "%Y-%m-%d").strftime("%U")

train_data['month_key'] = train_data['Date'].map(f)
train_data['year'] = train_data['Date'].map(year_f)
train_data['week'] = train_data['Date'].map(year_w)

n = 1
    
grouped = train_data.groupby(['Trap','year'])

trapLines = None

curr_trap = ""
for (trap,year), group in grouped:       
    plt.plot(group["NumMosquitos"])
   