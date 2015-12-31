import numpy as np
import pandas as pd
import re
import csv

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