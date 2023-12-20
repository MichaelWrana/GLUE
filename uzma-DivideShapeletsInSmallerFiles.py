import pandas as pd
import stumpy
import numpy as np
import datetime as dt
import random
import math
import pickle
import sys
import os

from statistics import mean
from tqdm.auto import tqdm
from multiprocessing import Pool

import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

fld_Source = '../results/IPT/shapelets_num_random_seed_v3/'
fld_Destination = '../results/IPT/shapelets_num_random_seed_v3/smaller_shapelet_files/'

train_ratio = 0.9

experiments = range(0, 5)
coeff = 0.25
p = 1
q = 1
r = 1

parts = 15

for exp  in experiments:
    
    filename = fld_Source + 'exprmnt=' + str(exp) + \
                'shapelet_size=' + str(coeff) + 'p'+ str(p) + 'q' + str(q) + 'r' + str(r)
    if os.path.exists(filename) == True:
        print(filename,"exists")
    else:
        break
    

    with open(filename, 'rb') as f:
        shapelets = pickle.load(f)
     
    writtenshapelets = 0
    print(len(shapelets))
    offset = int(len(shapelets)/parts)
    for i in range(0, parts-1):
        #print("i*100",i*100,"(i+1)*100",(i+1)*100)
        fewer_shapelets = shapelets[i*offset:(i+1)*offset]
        fn = fld_Destination + 'exp'+ str(exp) + '/exprmnt=' + str(exp) + \
                'shapelet_size=' + str(coeff) + 'p'+ str(p) + 'q' + str(q) + 'r' + str(r) + "_part" + str(i)
        with open(fn, 'wb') as f:
            pickle.dump(fewer_shapelets, f) 
        writtenshapelets += len(fewer_shapelets)
        #print("writtenshapelets",writtenshapelets)
            
    #print("(i+1)*100",(i+1)*100)
    
    fewer_shapelets = shapelets[(i+1)*offset:]
    fn = fld_Destination + 'exp'+ str(exp) + '/exprmnt=' + str(exp) + \
            'shapelet_size=' + str(coeff) + 'p'+ str(p) + 'q' + str(q) + 'r' + str(r) + "_part" + str(i)
    with open(fn, 'wb') as f:
        pickle.dump(fewer_shapelets, f) 
    writtenshapelets += len(fewer_shapelets)  
    print("writtenshapelets",writtenshapelets,"\n\n")
    #break
        
        
    