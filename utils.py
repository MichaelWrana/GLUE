#!/usr/bin/env python
# coding: utf-8

# In[1]:


import stumpy
import numpy as np
import pickle

from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split



def distance_to_shapelet(data, shapelets):    
    # processed output data
    data_out = np.zeros((len(data),len(shapelets)))
    
    # loop over each sample in the dataset
    for i,sample in enumerate(tqdm(data)):
        shapelet_score = np.empty(len(shapelets))
        # for each shapelet, calculate distance and assign a score
        for j,shapelet in enumerate(shapelets):
            try:
                dist = stumpy.mass(shapelet, sample)
            except ValueError:
                dist = stumpy.mass(sample, shapelet)
            shapelet_score[j] = dist.min()
        data_out[i] = shapelet_score
    
    return data_out

def process_traces(shapelets, name):
    X, y = [], []

    # iterate over dictionary and re-format into X and y
    for trace_id, trace_vals in traces.items():
        for trace in trace_vals:
            X.append(trace)
            y.append(trace_id)
    
    print("Processing" + name + "... " + "(" + str(len(X)) + " traces)")
    
    # convert traces into float64 data type
    X = [np.asarray(trace).astype('float64') for trace in X]
    # clear empty trace values in data
    X = [trace[~np.isnan(trace)] for trace in X]    
    # compute distance between input trace and shapelet arrays
    # return as new X
    X = distance_to_shapelet(X, shapelets)
    
    return X, y

def evaluate_parameters(namestring):
    
    print(namestring)
    
    files = {
        'shapelets': folder_shapelets + namestring,
        'X': folder_X + namestring,
        'y': folder_y + namestring
    }
    try:
        with open(files['shapelets'], 'rb') as f:
            shapelets = pickle.load(f)
    except FileNotFoundError:
        print("Shapelet File Missing:" + files['shapelets'] + ", skipping...")
        return
    
    shapelets = [shapelet.astype('float64') for shapelet in shapelets]
    
    X, y = process_traces(shapelets, files['shapelets'])
    
    with open(files['X'], 'wb') as f:
        pickle.dump(X, f)
        
    with open(files['y'], 'wb') as f:
        pickle.dump(y, f)

global traces
with open("../datasets/ds19.npy", 'rb') as f:
    traces = pickle.load(f)

global folder_scores
folder_scores = "../results/scores/"
global folder_shapelets
folder_shapelets = "../results/shapelets/"
global folder_X
folder_X = "../results/data/X/"
global folder_y
folder_y = "../results/data/y/"