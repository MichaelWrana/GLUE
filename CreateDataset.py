#!/usr/bin/env python
# coding: utf-8

# In[2]:


import stumpy
import pickle
import numpy as np
from tqdm import tqdm
import sklearn.metrics as metrics

from multiprocessing import Pool
from sklearn.ensemble import RandomForestClassifier
from pipelinetools import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[3]:


# LOAD AND TRAIN THE MODEL


# In[4]:


traces_train = load_traces('../transformer/transformer_train')
traces_test = load_traces('../transformer/transformer_test')
traces_kfp = load_traces('../transformer/transformer_kfp')



# In[5]:


def merge_traces(traces_for_merge, overlap_factor):
    signs = [np.sign(trace) for trace in traces_for_merge]

    overlap_val = []
    
    for i in range(len(traces_for_merge)-1):
        overlap_index = (round(len(traces_for_merge[i]) * (1-overlap_factor)))
        if overlap_index ==  len(traces_for_merge[i]):
            overlap_index -= 1
        
        overlap_val.append(abs(traces_for_merge[i][overlap_index]))
    
    Total_overlap = 0
    for i in range(1,len(traces_for_merge)):
        Total_overlap += overlap_val[i-1]
    
        traces_for_merge[i] = [(abs(traces_for_merge[i][k]) + Total_overlap) * signs[i][k]\
                           for k in range(len(traces_for_merge[i]))]
        
            
    
    
    
    combined = np.concatenate(traces_for_merge)
    
    #print('combined before sorting',combined)
    #print('combined after sorting',sorted(combined, key=abs))
    
#     fig,ax = plt.subplots(2)
#     ax[0].plot(combined)
#     #plt.close()
#     #plt.figure()
#     x  = sorted(combined, key=abs)
#     print(len(x))
#     ax[1].plot(x)
    
    
    
    return sorted(combined, key=abs)


# In[6]:


def GetYbitvector(ClassesConsidered, y):
    
    lst_Y_bit = []
    
    for i in range(len(y)):
        Y_bit = np.zeros((ClassesConsidered+1))
        for j in y[i]:
            if j == -1:
                Y_bit[-1] = 1
            else:
                Y_bit[j] = 1
        lst_Y_bit.append(Y_bit)
    lst_Y_bit = np.array(lst_Y_bit)
    return lst_Y_bit
    


# In[7]:


def ToNumpyArray(list_2d):
 
    
    
    # Find the maximum length of all rows
    max_length = max(len(row) for row in list_2d)

    # Pad each row with zeros to make them all the same length
    padded_list_2d = [row + [0] * (max_length - len(row)) for row in list_2d]

    # Convert the padded 2D list to a NumPy array
    array_2d = np.array(padded_list_2d)

    
    return array_2d


# In[8]:


def CreateMergeTraces(NumTraces,NumClassPerTrace,overlap_factor,traces,ClassesConsidered,bIPT=True):
    X = []
    y = []
    X_ids = []

     
    
    for i in tqdm(range(NumTraces)):
        traces_for_merge = []
        class_id = []
        traces_id = []
        
        
        
        
        mon = random.randint(0,NumClassPerTrace)
        
        for j in range(NumClassPerTrace):
            
            #Add one monitored class and rest unmonitored
            if j == mon:
                true_id = random.randint(0,ClassesConsidered-1)

                trace_id = random.randint(0,len(traces[true_id])-1)

                traces_for_merge.append(traces[true_id][trace_id])

                class_id.append(true_id)

                traces_id.append(trace_id)
            
            # Now adding unmonitored classes
            else:
                
                true_id = -1

                trace_id = random.randint(0,len(traces[true_id])-1)

                traces_for_merge.append(traces[true_id][trace_id])

                class_id.append(true_id)
                traces_id.append(trace_id)


        combo_trace = merge_traces(traces_for_merge, overlap_factor)
        
#        l = [len(x) for x in traces_for_merge]
#        print(l,np.sum(l),len(combo_trace))
        
#         if bIPT == True:
#             combo_trace = convert_to_ipt(combo_trace)

                                    
        X.append(combo_trace)
        X_ids.append(traces_id)
#        print(traces_id,class_id)
        y.append(class_id)

    print(len(X))
    
    return ToNumpyArray(X),y,X_ids


# In[ ]:


import os
from sklearn.model_selection import train_test_split


NumTraces = 10000
overlap_factor = 0
bIPT = False
NumClassPerTrace = 4
ClassesConsidered = 50

seed = 1018

outpath = "datasets/processed/my_open_4tab"

X,y,_ = CreateMergeTraces(NumTraces,NumClassPerTrace,overlap_factor,traces_train,ClassesConsidered,bIPT)

print(type(X))
print(type(X[0]))

lst_Y_bit = GetYbitvector(ClassesConsidered, y)
#print(lst_Y_bit)
y = lst_Y_bit
print(type(y))
print(type(y[0]))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=seed)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, train_size=0.9, random_state=seed)
print(f"Train: X = {X_train.shape}, y = {y_train.shape}")
print(f"Valid: X = {X_valid.shape}, y = {y_valid.shape}")
print(f"Test: X = {X_test.shape}, y = {y_test.shape}")

np.savez(os.path.join(outpath, "train.npz"), X = X_train, y = y_train)
np.savez(os.path.join(outpath, "valid.npz"), X = X_valid, y = y_valid)
np.savez(os.path.join(outpath, "test.npz"), X = X_test, y = y_test)


# In[ ]:




