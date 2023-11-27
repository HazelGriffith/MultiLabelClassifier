# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 15:11:18 2023

@author: hazel
"""
import numpy as np
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.ensemble import AdaBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

start_time = time.time()

samples = {}


# This first section loads the class labels and feature sets into dictionaries
sampleClasses = {}
with open("labels_train.csv") as classLabelsFile:
    classLabelsFile.readline()
    for i, line in enumerate(classLabelsFile):
        temp = []
        classes = line.split(",")
        temp.extend(list(classes))
        temp = [float(j) for j in temp]
        sampleClasses.update({i: temp})

r1Samples = {}
with open("R1_train.csv") as R1_trainFile:
    R1_trainFile.readline()
    for i, line in enumerate(R1_trainFile):
        temp = []
        r1Data = line.split(",")
        temp.extend(list(r1Data))
        temp = [float(j) for j in temp]
        r1Samples.update({i: temp})

r2Samples = {}
with open("R2_train.csv") as R2_trainFile:
    R2_trainFile.readline()
    for i, line in enumerate(R2_trainFile):
        temp = []
        r2Data = line.split(",")
        temp.extend(list(r2Data))
        temp = [float(j) for j in temp]
        r2Samples.update({i: temp})

r3Samples = {}
with open("R3_train.csv") as R3_trainFile:
    R3_trainFile.readline()
    for i, line in enumerate(R3_trainFile):
        temp = []
        r3Data = line.split(",")
        temp.extend(list(r3Data))
        temp = [float(j) for j in temp]
        r3Samples.update({i: temp})

r4Samples = {}
with open("R4_train.csv") as R4_trainFile:
    R4_trainFile.readline()
    for i, line in enumerate(R4_trainFile):
        temp = []
        r4Data = line.split(",")
        temp.extend(list(r4Data))
        temp = [float(j) for j in temp]
        r4Samples.update({i: temp})

r5Samples = {}
with open("R5_train.csv") as R5_trainFile:
    R5_trainFile.readline()
    for i, line in enumerate(R5_trainFile):
        temp = []
        r5Data = line.split(",")
        temp.extend(list(r5Data))
        temp = [float(j) for j in temp]
        r5Samples.update({i: temp})

r6Samples = {}
with open("R6_train.csv") as R6_trainFile:
    R6_trainFile.readline()
    for i, line in enumerate(R6_trainFile):
        temp = []
        r6Data = line.split(",")
        temp.extend(list(r6Data))
        temp = [float(j) for j in temp]
        r6Samples.update({i: temp})

samples.update({"R1": r1Samples, "R2": r2Samples, "R3": r3Samples,
               "R4": r4Samples, "R5": r5Samples, "R6": r6Samples})


loading_time = time.time()-start_time
print("Loading Feature Sets Complete in: " + str(loading_time))

# This next section performs 5-fold CV for the oneVsRestClassifier using
# GaussianNB across all feature sets
start_time = time.time()

cv_scores_mean_perR = {}
fmicro_average = make_scorer(f1_score, average='micro')
Y = list(sampleClasses.values())

# for key in list(samples.keys()):

X = list(r2Samples.values())

pipe = Pipeline(
    [('sampl', SMOTE()), ('clf', AdaBoostClassifier(GaussianNB(), n_estimators=100, random_state=42))])
GNBC = OneVsRestClassifier(pipe, n_jobs=2)
cv_scores = cross_val_score(
    GNBC, X, Y, scoring=fmicro_average, cv=5, n_jobs=2)

print(np.mean(cv_scores))

'''
cv_scores_mean_perR.update({key: np.mean(cv_scores)})

running_time = time.time()-start_time
print("Running Classifier Complete in: " + str(running_time))

for key in list(samples.keys()):
    print(key + ": ")
    print(cv_scores_mean_perR.get(key))
'''
