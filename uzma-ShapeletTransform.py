#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import stumpy
import numpy as np
import datetime as dt
import random
import math
import pickle
import sys
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


# In[114]:


'''
Compute the minimum distance beteen data samples and shapelets
Input:
    data = list of individual packet traces
    shapelets = list of shapelets
Output:
    minimum distance between each sample in data compared with each sample in shapelet
    shape = (len(data),len(shapelets))
'''
def distance_to_shapelet(data, shapelets):
    #data = np.asarray(data)
    #print(len(data))
    
    # processed output data
    data_out = np.zeros((len(data),len(shapelets)))
    
#         X[idx] = None
#         y[idx] = None
#     X = [trace for trace in X if trace is not None]
#     y = [value for value in y if value is not None]
    

    # loop over each sample in the dataset
    for i,sample in enumerate(tqdm(data)):
        shapelet_score = np.empty(len(shapelets))
        # for each shapelet, calculate distance and assign a score
        for j,shapelet in enumerate(shapelets):
            if len(sample) < len(shapelet):
                dist = stumpy.mass(sample,shapelet)
                shapelet_score[j] = dist.min()
            else:
                dist = stumpy.mass(shapelet, sample)
                shapelet_score[j] = dist.min()
        data_out[i] = shapelet_score
    
    return data_out

'''
Computes distances between input samples and shapelets, returns X for classifier
Also cleans data and ensures no random errors due to length, NaN, etc...
Underlying function that performs comparison is distance_to_shapelet
Selects data samples (with replacement)
note: some samples will always be bad so actual length of X is less

Input:
    num_traces = numner of traces to process
    save = save output to file
    filenames = tuple that represents (name of X file, name of y file)

Output:
    X values for classifier of shape (None, 100)
    y values for classifier of shape (None, )
'''
                   
def process_traces(num_traces,traces, shapelets, save=True, filenames=("X.pkl","y.pkl")):
    
    X, y = [], []
        
    idx = list(traces.keys())
    traces_list = range(0,len(traces[0]))
    #print(idx, traces_list)

    parameter_list = np.array(np.meshgrid(idx,traces_list)).T.reshape(-1,2)
    #print(parameter_list.shape,parameter_list)

    pl2 = []
    for pl in parameter_list:
        pl2.append(pl)
    
    
    random_traces = random.sample(pl2,num_traces)
    #print(random_traces)

    for i in range(0,len(random_traces)):
        #print("random_traces[i][0]",random_traces[i][0],"\trandom_traces[i][1]",random_traces[i][1])
        X.append(traces[random_traces[i][0]][random_traces[i][1]])
        y.append(random_traces[i][0])


    # process and remove useless entries (too short)
    X = [np.asarray(trace).astype('float64') for trace in X]
    
    
    X = [trace[~np.isnan(trace)] for trace in X]    
    # what about these Y's?
    
#     removals = [i for i,x in enumerate(X) if len(x) < shapelet_size]
#     for idx in removals:
#         X[idx] = None
#         y[idx] = None
#     X = [trace for trace in X if trace is not None]
#     y = [value for value in y if value is not None]

    # compute distance between input trace and shapelet arrays
    # return as new X

    X = distance_to_shapelet(X, shapelets)
    
    if save:
        with open(filenames[0], 'wb') as f:
            pickle.dump(X, f)

        with open(filenames[1], 'wb') as f:
            pickle.dump(y, f)
    
    return X, y




# In[115]:


'''
Utility function for pipeline of evaluating different grid search parameters
Output: a new file located in ../results/param1-val1_param2-val2_param3-val3
        the file contains a pickled python object
        with the scores for top-1, top-3, and top-5 classifier accuracy
'''
# note: python multiprocessing is really annoying to work with
# function needs to be in a separate .py file which is imported
# and function can only have 1 argument
# list input which is immediately used for what would be the arguments
def evaluate_parameters(arr):
    
#     global train_traces
#     global test_traces
    
    
    train_ratio = float(arr[0])
    test_train_samples = int(arr[1])
#     experiments = int(arr[2])
#     coeff = (arr[3])
#     p = int(arr[4])
#     q = int(arr[5])
#     r = int(arr[6])
    fn_shapelet = arr[2]
    
    print('\nRecieved','===fn_shapelet:' , fn_shapelet , train_ratio , test_train_samples)
    
    shapelets = ReadFromfile(fldr_Shapelet + fn_shapelet)
    if shapelets == []:
        return -1
    else:
        print("Read shapelets file", fldr_Shapelet + fn_shapelet)
    

    print("Processing Traces..."+ fn_shapelet[fn_shapelet.find('part'):])
    train_samples = int(test_train_samples * train_ratio)
    test_samples = int(test_train_samples - train_samples)
    
    
    X_train, y_train = process_traces(train_samples,train_traces,shapelets, False)
    X_test, y_test = process_traces(test_samples,test_traces,shapelets, False)
    
    
    
    Writetofile(fldr_Destination_X_y + 'X/X_train_' + fn_shapelet + "_samples_" + str(test_train_samples),X_train)
    Writetofile(fldr_Destination_X_y + 'y/y_train_' + fn_shapelet + "_samples_" + str(test_train_samples),y_train)
    Writetofile(fldr_Destination_X_y + 'X/X_test_' + fn_shapelet + "_samples_" + str(test_train_samples),X_test)
    Writetofile(fldr_Destination_X_y + 'y/y_test_' + fn_shapelet + "_samples_" + str(test_train_samples),y_test)
    


def ReadFromfile(filename):
    
    if os.path.exists(filename) == False:
        return []
    
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        
    return data
    


# In[117]:


def Writetofile(filename,trace):
    
    
    with open(filename, 'wb') as f:
        pickle.dump(trace, f) 
        
    return 1
    


# In[118]:


# SETUP
# In[86]:

import datetime

e = datetime.datetime.now()

print ("Current date and time = %s" % e)

print ("Today's date:  = %s/%s/%s" % (e.day, e.month, e.year))

print ("The time is now: = %s:%s:%s" % (e.hour, e.minute, e.second))

global train_traces
global test_traces

global fldr_Shapelet
global fldr_Destination_X_y

fldr_Shapelet = '../results/IPT/shapelets_num_random_seed_v3/smaller_shapelet_files/exp0/'
fldr_Destination_X_y = '../results/IPT/data/'
train_ratio = [0.9]
test_train_samples = [20000]
experiments = range(0, 5)
coeff = 0.25
p = 1
q = 1
r = 1


str_uniqueIdentifier = 'shapelet_size='+ str(coeff) + 'p' + str(p) + 'q' + str(q) + 'r' + str(r)

lst_ShapeletFiles = []

p = Path(fldr_Shapelet)
for fn in p.iterdir():
    if fn.is_dir() == True:
        continue
    if str_uniqueIdentifier in str(fn):
        fname = str(fn).split("/")[-1]
        lst_ShapeletFiles.append(fname)
        print(fname)
    
    


fn_traintrace = '../ipt_traces_train_'+str(train_ratio[0])+ '.npy' 

with open(fn_traintrace, 'rb') as f:
    train_traces = pickle.load(f)

fn_testtrace = '../ipt_traces_test_'+str(train_ratio[0])+ '.npy' 

with open(fn_testtrace, 'rb') as f:
    test_traces  = pickle.load(f)

# PART 2: Shapelet transformation



parameter_list = np.array(np.meshgrid(train_ratio, test_train_samples,lst_ShapeletFiles)).T.reshape(-1,3)
    
print(parameter_list)

parts = len(parameter_list)
with Pool(parts) as p:
    p.map(evaluate_parameters, parameter_list)

e = datetime.datetime.now()

print ("Current date and time = %s" % e)

print ("Today's date:  = %s/%s/%s" % (e.day, e.month, e.year))

print ("The time is now: = %s:%s:%s" % (e.hour, e.minute, e.second))