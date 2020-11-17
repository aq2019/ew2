import pandas as pd
import streamlit as st
import numpy as np


from models import lr_svm_rf
from collections import defaultdict

from load_data import tables, tagset, conversation, allTags, all_ids
from utilities import readGlove, split_data
from preprocess import text_to_wordlist, word_embedding, doc_embedding, get_data


'''
# TagTalk

_Emotion Recognition and Information Extraction from Discourse_

'''


emotion_labels = ['Anxiety/Stress', 
        'Confidence', 'Stuck', 'Undervalued', 'Fear', 
        'Frustrated', 'Uncertain', 'Bullied']

issue_labels = ['Benefits or Leave',  'Communication',
       'Discrimination', 'Harassment', 
       'Payroll or time issue', 'Performance',
       'Workload/Hours', 'Job Search']

people_labels = ['Manager', 'Coworker']

categories = [emotion_labels, people_labels, issue_labels]

ids = all_ids().get


text = st.text_area('Please describe your situation:', '''''')

if text != '':
    true_tags = defaultdict()
    false_tags = defaultdict()
    nd = [text_to_wordlist(text)]
    pred_one_data = defaultdict()
    for cat in categories:
        for lb in cat:
            n = logisticRegr(lb)
            n.load()
            pred_one_data[lb] = n.predict(new_data = nd)
    for t in pred_one_data.keys():
        if pred_one_data[t].tolist()[0] is True:
            true_tags[t] = True
        else:
            false_tags[t] = False
    st.write('The following tags are identified: ', true_tags)
    st.write('The following tags are not identified: ', false_tags)
    




