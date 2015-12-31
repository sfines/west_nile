__author__ = 'Steven Fines'

import numpy as np
import pandas as pd
import re
import csv
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.cross_validation import StratifiedKFold


from sklearn.pipeline import make_pipeline
import math

import matplotlib.pyplot as plt

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

def generate_importance_plot(cols, imp_features):
    fig = plt.figure(figsize=(6 * 1.618, 6))
    index = np.arange(len(cols))
    bar_width = 0.15
    plt.bar(index, imp_features, color='black', alpha=0.5)
    plt.xlabel('features')
    plt.ylabel('importance')
    plt.title('Feature importance')
    plt.xticks(index + bar_width, cols)
    plt.tight_layout()
    fig.autofmt_xdate()
    plt.show()

def generate_auc_features(scores):
    fig = plt.figure()
    x = np.linspace(0, len(scores))
    plt.xlabel('Tree Depth')
    plt.ylabel('AUC Score')
    plt.plot(scores[0], color='r', label="")
    plt.plot(scores[1], color='b')
    plt.show()


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, [0, 1], rotation=45)
    plt.yticks(tick_marks, [0, 1])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


#dict of known mosquito species
species_dict = {
    'CULEX RESTUANS': 1,
    'CULEX PIPIENS': 2,
    'CULEX PIPIENS/RESTUANS': 3,
    'CULEX TERRITANS': 4,
    'CULEX SALINARIUS': 5,
    'CULEX ERRATICUS': 6,
    'CULEX TARSALIS': 7,
    'UNSPECIFIED CULEX': 8
}


trap_re = re.compile("T(\d*)\w?")

def trapToId(x, trap_re):
    mtch = trap_re.match(x).group(1)
    return int(mtch)


cols = ['Station', 'Latitude', 'Longitude', 'weekNumber', 'Tmax', 'Tmin', 'Tavg', 'PrecipTotal',
                   'StnPressure', 'SeaLevel', 'ResultSpeed', 'ResultDir', 'AvgSpeed', 'DewPoint', 'Sunrise', 'Sunset', 'SevendaymedianTmp',
                   'ThreedaymedianTmp', 'SevendaymedianMinTmp', 'ThreedaymedianMinTmp', 'RelHumidity', 'TrapFactor', 'SpeciesFactor']

# Prepare the Data
df = pd.read_csv('../input/merged_data.csv', header=0,  parse_dates=True, infer_datetime_format=True)
df['WnvPresent'] = df['V2'].map(lambda x: regularize(x))
df['TrapId'] = df['TrapFactor'].map(lambda x: int(trap_re.match(x).group(1)))
df['SpeciesId'] = df['SpeciesFactor'].map(lambda x: species_dict.get(x, 9))
df['RelHumidity'] = df.apply(lambda x: relative_humidity(x['Tavg'], x['DewPoint']), axis=1)

df = df.drop(['WnvPresentChar', 'V2', 'V1', 'ID', 'Date', 'ReadingDate', 'SpeciesFactor', 'TrapFactor'], axis=1)

# Move the target column to the front of the data frame, where numpy and sklearn expect it.
predict_df = pd.DataFrame(df.pop('WnvPresent'))
predict_df = predict_df.join(df)

predict = predict_df.values

cv = StratifiedKFold(predict[0::, 0], n_folds=2, shuffle=True)
winner_fit = None
winner_score = 0.0
winner_kbest_features = None
winner_selected_features = None

auc_score_curve = []
auc_score_curve.append([])
auc_score_curve.append([])

for j in range(8, 20):
    print("Tree Depth: {0}".format(j))

    for i, (train, test) in enumerate(cv):
        # Create a Random forest with 500 estimators and use 4 cpu cores to do the job

        _forest = RandomForestClassifier(n_estimators=500,  max_depth=j, n_jobs=4, min_samples_leaf=50)
        #forest = GradientBoostingClassifier(n_estimators=500, init=_forest)
        forest = AdaBoostClassifier(n_estimators=100, base_estimator=_forest)
        clf = make_pipeline(forest)
        fit = clf.fit(predict[train, 1::], predict[train, 0])

        auc_score = roc_auc_score(predict[test, 0], fit.predict_proba(predict[test, 1::])[:, 1])
        print("Fold AUC score: {0}".format(auc_score))
        auc_score_curve[i].append(auc_score)

        if auc_score > winner_score:
            winner_score = auc_score
            winner_fit = fit
            winner_selected_features = forest.feature_importances_
            print(confusion_matrix(predict[test, 0], fit.predict(predict[test, 1::])))

# Feature Importance plot
# generate_importance_plot([cols[int(x)] for x in winner_kbest_features], winner_kbest_features)

# Generate Confusion Matrix
train_predictions = winner_fit.predict(predict[0::, 1::])
confusion = confusion_matrix(predict[0::, 0], train_predictions)
print("Winning Confusion Matrix: ")
print(confusion)

plot_confusion_matrix(confusion)

print(winner_score)
print(auc_score_curve)
generate_auc_features(auc_score_curve)
# Now we've trained the forest on our Training set, it's time to prepare the test data
# in the same manner so that we can get predictions

test_df = pd.read_csv('../input/query_result.csv', header=0)
test_df['TrapId'] = test_df['TrapFactor'].map(lambda x: trapToId(x, trap_re))
test_df['SpeciesId'] = test_df['SpeciesFactor'].map(lambda x: species_dict.get(x, 8))
test_df['RelHumidity'] = test_df.apply(lambda x: relative_humidity(x['Tavg'], x['DewPoint']), axis=1)

test_ids = test_df['Id'].values
test_df = test_df.drop(['ReadingDate', 'SpeciesFactor', 'TrapFactor', 'Id'], axis=1)



# test_df = test_df[cols]

output = winner_fit.predict_proba(test_df.values)
_file = open("predictions.csv", "w")
output_writer = csv.writer(_file)
output_writer.writerow(['Id', 'WnvPresent'])
output_writer.writerows(zip(test_ids, output[:, 1]))
_file.close()

print("Done.")
