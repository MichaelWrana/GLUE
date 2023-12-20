#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import stumpy
import numpy as np
import datetime as dt
import random
import math
import pickle
import sys
import os
import os
import os.path as path
from pathlib import Path

from statistics import mean
from tqdm.auto import tqdm
from multiprocessing import Pool

import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# In[5]:


'''
Evaluate performance of sklearn classifier on data samples - 90/10 training testing split

Input:
    clf: sklearn classifier object
    X: x values
    y: y values
    topk: k values for evaluation metrics
Output:
    list of length topk with accuracy for testing data
'''

def classifier_performance(clf, X_train, X_test, y_train, y_test, topk=[1,3,5]):
    
    
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)
    
    scores = []
    for k in topk:
        correct = 0
        for i in range(len(y_prob)):
            ind = np.argpartition(y_prob[i], -k)[-k:]
            if y_test[i] in ind:
                correct += 1
        scores.append(correct/len(y_prob))
    
    return scores


# In[ ]:


def getXY(fldr_source,str_uniqueIdentifier1,str_uniqueIdentifier2,str_uniqueIdentifier3):
    
    X = []
    
    p = Path(fldr_source)
    for fn in p.iterdir():
        if fn.is_dir() == False:
            filename = str(fn)
           
#             print(str_uniqueIdentifier1,str_uniqueIdentifier2,str_uniqueIdentifier3)
#             print(filename)

            if str_uniqueIdentifier1 in filename and str_uniqueIdentifier2 in filename and \
            str_uniqueIdentifier3 in filename:
                print(filename)
                with open(filename, 'rb') as f:
                    X = pickle.load(f)
                    return X
    return X


# In[ ]:

import datetime
e = datetime.datetime.now()
print ("Classifier start time= %s" % e)

## PART 3 : Model building and inference
train_ratio = 0.9
test_train_samples = 20000
experiments = range(0, 5)
coeff = 0.25
p = 1
q = 1
r = 1

fldr_source_X =  "../results/IPT/data/combineX/"
fldr_source_Y =  "../results/IPT/data/combineY/"
fldr_dest_score = "../results/IPT/scores/"

str_uniqueIdentifier1 = 'shapelet_size='+ str(coeff) + 'p' + str(p) + 'q' + str(q) + 'r' + str(r)
str_uniqueIdentifier2 = 'samples_' + str(test_train_samples)

p = Path(fldr_source_X)

X_train = getXY(fldr_source_X,str_uniqueIdentifier1,str_uniqueIdentifier2,"X_train")
X_test = getXY(fldr_source_X,str_uniqueIdentifier1,str_uniqueIdentifier2,"X_test")


y_train = getXY(fldr_source_Y,str_uniqueIdentifier1,str_uniqueIdentifier2,"y_train")
y_test = getXY(fldr_source_Y,str_uniqueIdentifier1,str_uniqueIdentifier2,"y_test")
    
print(X_train.shape)
print(len(y_train))
print(X_test.shape)
print(len(y_test))
      
    
outfile_name = fldr_dest_score + "_" + str_uniqueIdentifier1 + "_" + str_uniqueIdentifier2
print(outfile_name)


    
clf = RandomForestClassifier()
scores = classifier_performance(clf,X_train, X_test, y_train, y_test)
    
print(scores)

    
with open(outfile_name, 'wb') as f:
    pickle.dump(scores, f)

e = datetime.datetime.now()
print ("Classifier end time= %s" % e)
