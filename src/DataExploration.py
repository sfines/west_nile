__author__ = 'Steven Fines <dataknife@gmail.com>'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

train_data = pd.read_csv("..\\input\\train.csv",
                          parse_dates=True, infer_datetime_format=True)

weather_data = pd.read_csv("..\\input\\weather.csv",
                          parse_dates=True, infer_datetime_format=True)

spray_data = pd.read_csv("..\\input\\spray.csv",
                          parse_dates=True, infer_datetime_format=True)

f = lambda x: dt.datetime.strptime(x, "%Y-%m-%d").strftime("%Y-%m")
year_f = lambda x: dt.datetime.strptime(x, "%Y-%m-%d").strftime("%Y")
year_w = lambda x: dt.datetime.strptime(x, "%Y-%m-%d").strftime("%U")

train_data['month_key'] = train_data['Date'].map(f)
train_data['year'] = train_data['Date'].map(year_f)
train_data['week'] = train_data['Date'].map(year_w)


#mosquito_by_trap_date = train_data.groupby(["Trap", "Date"])["NumMosquitos", "WnvPresent"]
#
# plot = plt.figure()
# plot.savefig("test.png")
#
# distinct_traps = train_data["Trap"].drop_duplicates()
#
# distinct_year = train_data["year"].drop_duplicates()
#
# summarized_counts = train_data["Trap","year","week","NumMosquitos","WnvPresent"]

#create trap reference table
grouped_traps = train_data.groupby(["Trap","AddressNumberAndStreet","Latitude","Longitude"]).count().reset_index()
grouped_traps = grouped_traps.drop(["Date","Address","Species","Block","Street","AddressAccuracy","NumMosquitos","WnvPresent","month_key","year","week"], axis=1)
grouped_traps.to_csv("../input/traps.csv")


