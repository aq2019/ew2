import pickle
import numpy as np
from collections import defaultdict
import random
import pandas as pd

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

def completion_data():
    # return IDs for completion status classifier, return a dictionary 
    completed_id = all_ids().completed
    incomplete_id = all_ids().incomplete
    initiated_only_id = all_ids().initiated_only
    # find out those id that have more than one tags in [completed, incomplete, initiated_only]
    multi_tagged_comp_status = []
    for i in completed_id:
        if (i in incomplete_id) or (i in initiated_only_id):
            multi_tagged_comp_status.append(i)
    for i in incomplete_id:
        if i in initiated_only_id:
            multi_tagged_comp_status.append(i)
    # exclude conversations with multiple completion status tagging from the dataset; exlude ids with issue
    id_with_issue = [str(2388427861)]
    exclude = multi_tagged_comp_status + id_with_issue
    # remove multi-tagged data
    completion_data_ids = {}
    completion_data_ids['completed_id'] = [i for i in completed_id if i not in exclude]
    completion_data_ids['incomplete_id'] = [i for i in incomplete_id if i not in exclude]
    completion_data_ids['initiated_only_id'] = [i for i in initiated_only_id if i not in exclude]
    return completion_data_ids

class JoinShuffleDF():
    def __init__(self, df_list): # input a list of dataframes
        self.df = pd.concat(df_list, ignore_index=True)
        self.df = self.df.sample(frac=1).reset_index(drop=True).copy(deep=True)
    
    def split(self, label_type = 'initiated', train_ratio=0.8, drop_column = ['id', 'completion_status', 'init_indicator', 'completion_indicator']):
        # label_type: 'initiated' or 'completion'
        size = int(self.df.shape[0]*train_ratio)
        x = self.df.drop(drop_column, axis = 1).copy(deep=True)
        if label_type == 'initiated':
            y = self.df['init_indicator'].copy(deep=True)
        if label_type == 'completion':
            y = self.df['completion_indicator'].copy(deep=True)
        else:
            raise Exception('label_type should be initiated or completion')
        idx = self.df['id'].copy(deep=True)
        out = {}
        out['train_x'] = x.iloc[:size, :].copy(deep=True)
        out['val_x'] = x.iloc[size:, :].copy(deep=True)
        out['train_y'] = y[:size].copy(deep=True)
        out['val_y'] = y[size:].copy(deep=True)
        out['train_idx'] = idx[:size].copy(deep=True)
        out['val_idx'] = idx[size:].copy(deep=True)
        return out
        
        
        
def error_analysis(id_list):
    for i in id_list:
        print(i)
        print('actual tags: {0}'.format(allTags().byConvid(i)))
        print('------'*5)
        conversation(i).printExtract()
        print('======='*5)
        

def get_error(ytrue, ypred, idx, error_type = 'fp'):
    if error_type == 'fp':
        fp = []
        for i in range(len(ytrue)):
            if ytrue.values[i] == 0 and ypred[i] == 1:
                fp.append(idx.values[i])
        return fp
    if error_type == 'fn':
        fn = []
        for i in range(len(ytrue)):
            if ytrue.values[i] == 1 and ypred[i] == 0:
                fn.append(idx.values[i])
        return fn
    else:
        raise Exception('error_type should be fp or fn')
    


    
    
    

    


