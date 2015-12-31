from haversine import haversine
import pandas as pd
import numpy as np


def compute_distance(missing, known):
    return haversine(missing, known)*1000 #convert to meters

def chooseNearestTrap(missing_row, known_df):
    missing_tup = (missing_row['latitude'], missing_row['longitude'])
    temp_df = pd.DataFrame(index=known_df['trap'])

    vals = []
    for row in known_df.iterrows():
        target_tup = (row[1]['latitude'], row[1]['longitude'])
        dist = compute_distance(missing_tup, target_tup)
        vals.append(dist)

    series = pd.Series(index=known_df['trap'], data=np.asarray(vals))

    temp_df['trap'] = known_df['trap']
    temp_df['dist'] = series

    min_idx = temp_df['dist'].idxmin(axis=1)
    return min_idx


known_traps = pd.read_csv('..\\input\\traps.csv', header=0)
unknown_traps = pd.read_csv('..\\input\\missing_traps.csv', header=0)

unknown_traps['NearestKnown'] = unknown_traps.apply(lambda x: chooseNearestTrap(x, known_traps), axis=1)
unknown_traps.to_csv('..\\input\\missing_assignments.csv')