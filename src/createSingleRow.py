__author__ = 'Steven Fines'

import numpy as np
import pandas as pd
import re
import csv
import math


def regularize(x):
    if x > 0:
        return 1
    else:
        return 0

def map_species(x, species):
    if x['SpeciesFactor'].lower() == species.lower():
        return 1
    else:
        return 0

def to_celcius(x):
    return (x-32)*(5/9)

def to_fahrenheit(x):
    return ((9/5)*x)+32

def relative_humidity(tempInF, dewpoint):
    dp = to_celcius(dewpoint)
    tmp = to_celcius(tempInF)
    return 100*(math.exp((17.625*dp)/(243.04+dp))/math.exp((17.625*tmp)/(243.04+tmp)))


cols = ['TrapFactor','Station', 'Date','Latitude', 'Longitude', 'weekNumber', 'Tmax', 'Tmin', 'Tavg', 'PrecipTotal',
                   'StnPressure', 'SeaLevel', 'ResultSpeed', 'ResultDir', 'AvgSpeed', 'DewPoint', 'Sunrise', 'Sunset', 'SevendaymedianTmp',
                   'ThreedaymedianTmp', 'SevendaymedianMinTmp', 'ThreedaymedianMinTmp',   'RelHumidity']

df = pd.read_csv('../input/merged_data.csv', header=0,  parse_dates=True, infer_datetime_format=True)
df['WnvPresent'] = df['V2'].map(lambda x: regularize(x))
df['RelHumidity'] = df.apply(lambda x: relative_humidity(x['Tavg'], x['DewPoint']), axis=1)

df['RESTUANS'] = df.apply(lambda x: map_species(x, 'CULEX RESTUANS'), axis=1)
df['PIPENS'] = df.apply(lambda x: map_species(x, 'CULEX PIPIENS'), axis=1)
df['PIPIENS-RESTUANS'] = df.apply(lambda x: map_species(x, 'CULEX PIPIENS/RESTUANS'), axis=1)
df['TERRITANS'] = df.apply(lambda x: map_species(x, 'CULEX TERRITANS'), axis=1)
df['SALINARIUS'] = df.apply(lambda x: map_species(x, 'CULEX SALINARIUS'), axis=1)
df['ERRATICUS'] = df.apply(lambda x: map_species(x, 'CULEX ERRATICUS'), axis=1)
df['TARSALIS'] = df.apply(lambda x: map_species(x, 'CULEX TARSALIS'), axis=1)
df = df.drop(['SpeciesFactor'], axis=1)

#Detailed Records
detailed = df.groupby(cols)["V1", "WnvPresent", "RESTUANS","PIPENS","PIPIENS-RESTUANS","TERRITANS","SALINARIUS","ERRATICUS","TARSALIS"].sum().add_suffix('_detailed').reset_index()

detailed.to_csv("expanded.csv")

summaryCols = ['Date', 'weekNumber']

summary = df.groupby(summaryCols)["V1", "WnvPresent"].sum().add_suffix('_summary').reset_index()

summary.to_csv("totals.csv")

merged = pd.merge(summary, detailed, on=['Date', 'weekNumber'], how='inner')
merged['odds'] = merged.apply(lambda x: x['V1_detailed']/x['V1_summary'], axis=1)

merged.to_csv("merged.csv")

