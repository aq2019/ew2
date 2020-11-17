

import pickle
import numpy as np
from collections import defaultdict
import random

from load_data import tables, tagset, conversation, allTags, all_ids

def readGlove():
    # read the GloVe word embedding 'glove.6B.50d.txt' and create a dictionary {'word': [50 dimension np array word embedding]}
    glove50d_dict = defaultdict()
    with open('glove.6B.50d.txt','r') as f:
        for line in f:
            key, v = line_to_dict_entry(line)
            glove50d_dict[key] = v
        else:
            print('finished reading glove.6B.50d.txt')
    glove50d_dict = add_unknown_word_entry(glove50d_dict)
    filename = 'glove50d.sav'
    pickle.dump(glove50d_dict, open(filename, 'wb'))
    

def line_to_dict_entry(st):
    # help function for readGlove; extract an entry of dictionary {'word':vector} from a line in the txt file 
    ll = st.split(' ')
    ll[-1] = ll[-1][:-1] # delete \n at the end of line
    key = ll[0]
    v = np.asarray([float(x) for x in ll[1:]])
    return key, v

def add_unknown_word_entry(key_array_dict):
    # help function for readGlove
    # create an 'ukn' entry; assign its value as the average of 1000 random sample vectors from the dict
    # input: a dictionary with words as keys and wording embedding vectors as values
    sample_size = 1000
    rs = np.asarray(random.sample(list(key_array_dict.values()), sample_size))
    v = rs.sum(axis=0)
    key_array_dict['ukn'] = v
    return key_array_dict

def split_data(ids, X, y, training_ratio):
    # input - X: list of word lists; y: numpy array of boolean values; training_size: the proportion of data for training
    # output - ids_train, ids_test, X_train, y_train, X_test, y_test
    # data points with True value for y should be splited proportionally between training and text data
    idsa = np.asarray(ids)
    Xa = np.asarray(X)
    ya = np.asarray(y)
    ids0 = idsa[y==False]
    ids1 = idsa[y==True]
    X_0 = Xa[y==False]
    X_1 = Xa[y==True]
    y_0 = y[y==False]
    y_1 = y[y==True]
    training_size = int(X_0.shape[0]*training_ratio)
    ind = np.random.choice(X_0.shape[0], size = training_size, replace = False)
    qq = np.array([True]*X_0.shape[0])
    qq[ind] = False
    ids_0_train = ids0[ind]
    ids_0_test = ids0[qq]
    X_0_train = X_0[ind]
    X_0_test = X_0[qq]
    y_0_train = y_0[ind]
    y_0_test = y_0[qq]
    training_size1 = int(X_1.shape[0]*training_ratio)
    ind1 = np.random.choice(X_1.shape[0], size = training_size1, replace = False)
    qq1 = np.array([True]*X_1.shape[0])
    qq1[ind1] = False
    ids_1_train = ids1[ind1]
    ids_1_test = ids1[qq1]
    X_1_train = X_1[ind1]
    X_1_test = X_1[qq1]
    y_1_train = y_1[ind1]
    y_1_test = y_1[qq1]
    ids_train = np.concatenate((ids_0_train, ids_1_train), axis = 0)
    ids_test = np.concatenate((ids_0_test, ids_1_test), axis = 0)
    X_train = np.concatenate((X_0_train, X_1_train), axis = 0)
    X_test = np.concatenate((X_0_test, X_1_test), axis = 0)
    y_train = np.concatenate((y_0_train, y_1_train), axis = 0)
    y_test = np.concatenate((y_0_test, y_1_test), axis = 0)
    
    shuffle = np.random.choice(X_train.shape[0], size = X_train.shape[0], replace = False)
    ids_train = ids_train[shuffle]
    X_train = X_train[shuffle]
    y_train = y_train[shuffle]
    
    return ids_train, ids_test, X_train.tolist(), X_test.tolist(), y_train, y_test


    

    
    
    

    

