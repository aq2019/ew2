import pickle
import pandas as pd
import numpy as np
import random
import sklearn
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from collections import defaultdict

from load_data import tables, tagset, conversation, allTags, all_ids
from utilities import readGlove, split_data
from preprocess import text_to_wordlist, word_embedding, doc_embedding, get_data


class lr_svm_rf():
    def __init__(self, lb, method = 'LogisticRegression'):
        self.label = lb
        self.data = None
        self.model = None
        self.d2v_model = None
        self.tfidf_weighted_glove_docvec = None
        self.glove_embedding = word_embedding('GloVe') # get w2v as input for d2v - glove method
        self.method = method
                                       
    def split(self, conv_ids, training_ratio = 0.8):
        ids, X, y = get_data(conv_ids, self.label)
        data = defaultdict()
        data['ids_train'], data['ids_test'], data['X_train'], data['X_test'], data['y_train'], data['y_test'] = split_data(ids, X, y, training_ratio)
        self.data = data # update class feature!!
            
    def train(self, cost_weight = 'balanced'):
        # To change cost_weigth, replace input with a dictionary assigning weights to class labels. e.g., cost_weight = {True: 1.0, False:0.2}
        X_train = self.data['X_train']
        self.d2v_model = doc_embedding() # update class feature!!
        wts = self.d2v_model.get_tfidf(X_train, X_train) # get weights for d2v - glove method
        x_train = self.d2v_model.w2v_to_d2v(self.glove_embedding.w2v, wts)
        self.tfidf_weighted_glove_docvec = x_train # update class feature: document embedding from tfidf weighted average glove
        if self.method == 'LogisticRegression':           
            logisticRegr = LogisticRegression(class_weight=cost_weight, max_iter=1000, C = 1.5, solver = 'lbfgs')
            logisticRegr.fit(x_train, self.data['y_train'])
            self.model = logisticRegr # udpate class feature!!
            print('Finished training {0} model for label {1} with class weight {2}'.format(self.method, self.label, cost_weight))
        if self.method == 'SVM':
            svm_clf = svm.SVC(kernel='linear', C=1, class_weight = cost_weight).fit(x_train, self.data['y_train'])
            self.model = svm_clf # udpate class feature!!
            print('Finished training {0} model for label {1} with class weight {2}'.format(self.method, self.label, cost_weight))
        if self.method == 'RandomForest':
            rf = RandomForestClassifier(criterion = 'entropy', n_estimators = 100, class_weight = cost_weight, max_depth = 10, random_state = 0)
            rf.fit(x_train, self.data['y_train'])
            self.model = rf # udpate class feature!!
            print('Finished training {0} model for label {1} with class weight {2}'.format(self.method, self.label, cost_weight))
        
    def get_cv_score(self, fold = 5):
        scores_f1 = cross_val_score(self.model, self.tfidf_weighted_glove_docvec, self.data['y_train'], cv=fold, scoring = 'f1_macro')
        print('5-fold cross validation f1 score: {0}'.format(scores_f1))
        return scores_f1
    
    def predict(self, new_data = None):
        if new_data is not None: # new_data has to be list of word lists
            new_data_tfidf = self.d2v_model.get_tfidf(new_data)
            new_data_vec = self.d2v_model.w2v_to_d2v(self.glove_embedding.w2v, new_data_tfidf)
            new_data_pred = self.model.predict(new_data_vec)
            return new_data_pred
        
        test_tfidf = self.d2v_model.get_tfidf(self.data['X_test'])
        x_test = self.d2v_model.w2v_to_d2v(self.glove_embedding.w2v, test_tfidf)
        pred = self.model.predict(x_test)
        score = self.model.score(x_test, self.data['y_test'])        
        cm = metrics.confusion_matrix(self.data['y_test'], pred)
        
        # get ids in the confusion matrix
        mask_pred = np.array([True]*len(pred))
        mask_pred[pred] = False
        mask_y_test = np.array([True]*len(self.data['y_test']))
        mask_y_test[self.data['y_test']] = False
        tp = np.concatenate((pred.reshape(-1, 1),self.data['y_test'].reshape(-1, 1)), axis = 1).all(axis = 1)
        tn = np.concatenate((mask_pred.reshape(-1, 1),mask_y_test.reshape(-1, 1)), axis = 1).all(axis = 1)
        fn = np.concatenate((mask_pred.reshape(-1, 1),self.data['y_test'].reshape(-1, 1)), axis = 1).all(axis = 1)
        fp = np.concatenate((pred.reshape(-1, 1),mask_y_test.reshape(-1, 1)), axis = 1).all(axis = 1)
        ids = defaultdict()
        ids['true_positive_id'] = self.data['ids_test'][tp]
        ids['true_negative_id'] = self.data['ids_test'][tn]
        ids['false_positive_id'] = self.data['ids_test'][fp]
        ids['false_negative_id'] = self.data['ids_test'][fn]
        
        
        print('Predictions: {0}'.format(pred))
        print('Accuracy score: {0}'.format(score))
        print('Confusion matrix: {0}'.format(cm))
        return pred, score, cm, ids
           
    def save(self):
        pickle.dump(self.model, open('./models/{0}_model_{1}.sav'.format(self.method, self.label[0:4]), 'wb'))
        pickle.dump(self.d2v_model, open('./models/d2v_model_{0}.sav'.format(self.label[0:4]), 'wb'))
        pickle.dump(self.glove_embedding, open('./models/glove_embedding', 'wb'))
        pickle.dump(sell.tfidf_weighted_glove_docvec, open('./models/tfidf_weighted_glove_docvec', 'wb'))
        open('./models/{0}_model_{1}.sav'.format(self.method, self.label[0:4]), 'wb').close()
        open('./models/d2v_model_{0}.sav'.format(self.label[0:4]), 'wb').close()
        open('./models/tfidf_weighted_glove_docvec', 'wb').close()
       
        
    def load(self):
        self.model = pickle.load(open('./models/{0}_model_{1}.sav'.format(self.method, self.label[0:4]), 'rb'))
        self.d2v_model = pickle.load(open('./models/d2v_model_{0}.sav'.format(self.label[0:4]), 'rb'))
        self.glove_embedding = pickle.load(open('./models/glove_embedding', 'rb'))
        self.tfidf_weighted_glove_docvec = pickle.load(open('./models/tfidf_weighted_glove_docvec', 'rb'))
        
        open('./models/{0}_model_{1}.sav'.format(self.method, self.label[0:4]), 'rb').close()
        open('./models/d2v_model_{0}.sav'.format(self.label[0:4]), 'rb').close()
        open('./models/glove_embedding', 'rb').close()
        open('./models/tfidf_weighted_glove_docvec', 'rb').close()