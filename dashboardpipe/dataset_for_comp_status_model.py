from load_data import tables, conversation, stringToTimestamp, all_ids, allTags, db, tagset
from utilities import split_data
from preprocess import get_data
from nontext_feature_preprocessing import NontextFeature
from nontext_feature import feature

import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta, date

def create_df():

    completed_id = all_ids().completed
    incomplete_id = all_ids().incomplete
    initiated_only_id = all_ids().initiated_only
    compl_status_untagged_id = all_ids().compl_status_untagged

    # find out those id that have more than one tags in [completed, incomplete, initiated_only]
    multi_tagged_comp_status = []
    for i in completed_id:
        if (i in incomplete_id) or (i in initiated_only_id):
            multi_tagged_comp_status.append(i)
    for i in incomplete_id:
        if i in initiated_only_id:
            multi_tagged_comp_status.append(i)
            
    # clean 
    id_with_issue = [str(2388427861)]
    exclude = multi_tagged_comp_status + id_with_issue

    # remove multi-tagged data
    completed_id_clean = [i for i in completed_id if i not in exclude]
    incomplete_id_clean = [i for i in incomplete_id if i not in exclude]
    initiated_only_id_clean = [i for i in initiated_only_id if i not in exclude]

    # create nontext feature dataframe
    ntf_df_comp = NontextFeature(completed_id_clean).df
    ntf_df_incomp = NontextFeature(incomplete_id_clean).df
    ntf_df_init = NontextFeature(initiated_only_id_clean).df
    ntf_df_untagged = NontextFeature(compl_status_untagged_id).df
    adf = pd.concat([ntf_df_comp, ntf_df_incomp, ntf_df_init, ntf_df_untagged], ignore_index=True)
    adf = adf.sample(frac=1).reset_index(drop=True).copy(deep=True)
    for i in range(adf.shape[0]):
        if adf['completion_status'][i] == 'Incompl_init_only':
            adf.at[i, 'init_indicator'] = 1
        # if adf['completion_status'][i] == 'Completed' or adf['completion_status'][i] == 'Incomplete':
        else:
           adf.at[i, 'init_indicator'] = 0
        if adf['completion_status'][i] == 'Completed':
            adf.at[i, 'completion_indicator'] = 1
        # if adf['completion_status'][i] == 'Incomplete' or adf['completion_status'][i] == 'Incompl_init_only':
        else:
            adf.at[i, 'completion_indicator'] = 0
    return adf

class StatusDataset():
    def __init__(self, test_ratio = 0.05):
        self.df = create_df()
        self.init_y = self.df['init_indicator']
        self.init_x = self.df.drop(['id', 'start_time', 'end_time', 'completion_status', 'init_indicator', 'completion_indicator'], axis=1).copy() 
        self.comp_df = self.df[self.df['init_indicator'] == 0].copy()
        self.comp_y = self.comp_df['completion_indicator']
        self.comp_x = self.comp_df.drop(['id', 'start_time', 'end_time', 'completion_status', 'init_indicator', 'completion_indicator'], axis=1).copy()
        init_train_size = int(self.df.shape[0]*(1-test_ratio))
        comp_train_size = int(self.comp_df.shape[0]*(1 - test_ratio))
        self.init_train_x = self.init_x.iloc[:init_train_size, :].copy()
        self.init_train_y = self.init_y[:init_train_size]
        self.init_test_x = self.init_x.iloc[init_train_size:, :].copy()
        self.init_test_y = self.init_y[init_train_size:]
        self.comp_train_x = self.comp_x.iloc[:comp_train_size, :].copy()
        self.comp_train_y = self.comp_y[:comp_train_size]
        self.comp_test_x = self.comp_x.iloc[comp_train_size:, :].copy()
        self.comp_test_y = self.comp_y[comp_train_size:]
        
        
ds = StatusDataset()

print(ds.df.head())
for col in ds.df.columns:
    print(col)

print(ds.init_train_x.shape)
print(ds.init_test_x.shape)
print(ds.init_test_x.shape)
print(ds.init_test_y.shape)
print(ds.comp_train_x.shape)
print(ds.comp_train_y.shape)
print(ds.comp_test_x.shape)
print(ds.comp_test_y.shape)




    
    