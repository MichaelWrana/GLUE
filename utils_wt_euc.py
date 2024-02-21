#!/usr/bin/env python
# coding: utf-8

# In[1]:


import stumpy
import numpy as np
import pickle

from tqdm.auto import tqdm

from sklearn.model_selection import train_test_split

# def CalcWeightVector(length):
#     indices = np.array(range(1,length+1))
#     array_sum = np.sum(indices)
    
#     reverse_indices = length - indices + 1
    
#     WeightedArray = reverse_indices / array_sum
    
#     WeightedArrayLog = np.log(reverse_indices)
#     WeightedArrayLog = WeightedArrayLog/np.sum(WeightedArrayLog)
    
#     return WeightedArray,WeightedArrayLog

def CalcWeightVector(length,method):
    indices = np.array(range(1,length+1))
    array_sum = np.sum(indices)
    
    reverse_indices = length - indices + 1
    
    if method == "log":
        WeightedArray = np.log(reverse_indices)
        WeightedArray = WeightedArray/np.sum(WeightedArray)
    elif method == "linear":
        WeightedArray = reverse_indices / array_sum
    elif method == "inv_exp" :   
        WeightedArray = np.log(indices/array_sum)
        WeightedArray = WeightedArray/np.sum(WeightedArray)
    
    return WeightedArray
    
def weightedL2(a, b, w):
    q = a-b
    return np.sqrt((w*q*q).sum())

def SlideAndComputeDist(shorterSeq,LongerSeq,WtType):
    
    
    windowLength = len(LongerSeq) - len(shorterSeq)
    #print("len(LongerSeq)",len(LongerSeq), "len(shorterSeq)",len(shorterSeq))
    length = len(shorterSeq)
    
    W = CalcWeightVector(length,WtType)
     
    data_out = np.zeros((windowLength+1))
   
    for i in range(windowLength+1):
        data_out[i] = weightedL2(LongerSeq[i:length+i],shorterSeq[:length], W)
        #data_out[i] = np.linalg.norm(LongerSeq[i:length+i]- shorterSeq[:length])
        #data_out[i] = np.dot(LongerSeq[i:length+i],shorterSeq[:length])
        #print(i,":",data_out[i] )
     
    #print("len(data_out)",len(data_out),np.argsort(data_out))
    return np.min(data_out)
        
'''
Compute the minimum distance beteen data samples and shapelets
Input:
    data = list of individual packet traces
    shapelets = list of shapelets
Output:
    minimum distance between each sample in data compared with each sample in shapelet
    shape = (len(data),len(shapelets))
'''
def distance_to_shapelet_STUMPY_Replica(data, shapelets,WtType):
    
    # processed output data
    data_out = np.zeros((len(data),len(shapelets)))
   
    print(data_out.shape,len(data))

    # loop over each sample in the dataset
    for i,sample in enumerate(tqdm(data)):
        
#         print(len(sample),len(shapelets))
#         print("sample:",sample,"shapelets:",shapelets)

        shapelet_score = np.empty(len(shapelets))
        for j,shapelet in enumerate(shapelets):
        
            if len(sample) <= len(shapelet):
                shapelet_score[j] = SlideAndComputeDist(sample,shapelet,WtType)
                #print(sliding_dot_product(sample,shapelet))
            else:
                #print(sliding_dot_product(shapelet,sample))
                shapelet_score[j] = SlideAndComputeDist(shapelet,sample,WtType)
        data_out[i] = shapelet_score
        #break
    return data_out


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


from scipy import stats
def z_Transform(series):
    z_norm = []
    for i in range(0,len(series)):
        z_norm.append(stats.zscore(series[i]))
        
    return z_norm


def process_traces(shapelets, name,WtType):
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
    #X = distance_to_shapelet(X, shapelets)
    X = z_Transform(X)
    X = distance_to_shapelet_STUMPY_Replica(X, shapelets,WtType)
    
    return X, y

def evaluate_parameters(namestring):
    
    print(namestring,flush=True)
    
    Id = namestring[0] + "_part" + namestring[1]
    files = {
        'shapelets': folder_shapelets + "smaller_shapelet_files/" + Id,
        'X': folder_X + Id,
        'y': folder_y + Id
    }
    WtType = namestring[2]
    
    
    print("\ngot files", "files['shapelets']",files['shapelets'],"\tfiles['X']",files['X'], "\tfiles['y']:",files['y'])
    try:
        with open(files['shapelets'], 'rb') as f:
            shapelets = pickle.load(f)
    except FileNotFoundError:
        print("Shapelet File Missing:" + files['shapelets'] + ", skipping...")
        return
    
    shapelets = [shapelet.astype('float64') for shapelet in shapelets]
    
    shapelets = z_Transform(shapelets)
    
    X, y = process_traces(shapelets, files['shapelets'],WtType)
    
    with open(files['X'], 'wb') as f:
        pickle.dump(X, f)
        
    with open(files['y'], 'wb') as f:
        pickle.dump(y, f)
        
    print("written to files\tfiles['X']",files['X'], "\tfiles['y']:",files['y'], "\n*****\n")
    

global traces

fn_traces = 'ds19_negonly'

global traces
with open("../" + fn_traces + ".npy", 'rb') as f:
    traces = pickle.load(f)
    
global folder_shapelets
folder_shapelets = "../results/shapelets/" + fn_traces + "/"
 
fn_traces = fn_traces + "_wt_euc_invExp"
global folder_scores
folder_scores = "../results/scores/" + fn_traces + "/"
global folder_X
folder_X = "../results/data/" + fn_traces + "/X/"
global folder_y
folder_y = "../results/data/" + fn_traces + "/y/"

folder_data = "../results/data/" + fn_traces + "/"


# fn_traces = 'ds19_tamaraw_0210_1411'
# with open("../" + fn_traces + ".npy", 'rb') as f:
#     traces = pickle.load(f)

# global folder_scores
# folder_scores = "../results/scores/"
# global folder_shapelets
# folder_shapelets = "../results/shapelets/"
# global folder_X
# folder_X = "../results/data/X/"
# global folder_y
# folder_y = "../results/data/y/"