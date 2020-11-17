from nontext_feature import feature
from load_data import allTags

import pandas as pd

class NontextFeature():
    def __init__(self, id_list): #input a list of ids 
        self.id_list = id_list
        self.df = None
        self.create_feature() # get a dataframe with nontext features for the list of ids

    def create_feature(self):
        df = NontextFeature._create_row(self.id_list[0])
        if len(self.id_list) > 1:
            for i in range(1, len(self.id_list)):
                _temp_df = NontextFeature._create_row(self.id_list[i])
                df = pd.concat([df, _temp_df])
        self.df = df                      
    
    @staticmethod
    def _create_row(idx):
        ff = feature(idx)
        fdf = pd.DataFrame([[idx, ff.start_time, ff.end_time, ff.response_rate()[0], ff.response_rate()[1], ff.num_in_messages, ff.num_out_messages, ff.message_volume()[0], ff.message_volume()[1], ff.feedback(), ff.checkin(), ff.check_continue(), ff.come_back(), ff.thank(), ff.pop_up(), ff.conversation_last(), ff.our_best_to_you(), ff.resources(), ff.close_this(), None]], 
                           columns=['id', 'start_time', 'end_time', 'in_resp_rate', 'out_resp_rate', 'num_in_message', 'num_out_message', 'in_volume', 'out_volume', 'incl_feeback', 'incl_checkin', 'incl_continue', 'incl_comeback', 'incl_thank', 'incl_popup', 'incl_conversation_last', 'incl_our_best_to_you', 'incl_resources', 'incl_close', 'completion_status'])
        # add column 'completion_status'
        tags = allTags().byConvid(idx)
        compl_tags = ['1.1 Completed', '1.2 Incomplete', '1.3 Incomplete: Initiated only']
        if '1.1 Completed' in tags:
            fdf.at[0, 'completion_status'] = 'Completed'
        if '1.2 Incomplete' in tags:
            fdf.at[0, 'completion_status'] = 'Incomplete'
        if '1.3 Incomplete: Initiated only' in tags:
            fdf.at[0, 'completion_status'] = 'Incompl_init_only'
        if not list_contains(compl_tags, tags):
            fdf.at[0, 'completion_status'] = 'no_compl_status_tag'
        # add column 'init_indicator'
        if fdf['completion_status'][0] == 'Incompl_init_only':
            fdf.at[0, 'init_indicator'] = 1
        else: 
            fdf.at[0, 'init_indicator'] = 0
        # add column 'completion_indicator'
        if fdf['completion_status'][0] == 'Completed':
            fdf.at[0, 'completion_indicator'] = 1
        else:
            fdf.at[0, 'completion_indicator'] = 0
        
        return fdf
    
class CompletionStatusData():
    def __init__(self, df_list): # input a list of dataframes
        self.df = pd.concat(df_list, ignore_index=True)
        self.df = self.df.sample(frac=1).reset_index(drop=True).copy(deep=True)
        
        
def list_contains(List1, List2): 
  
    set1 = set(List1) 
    set2 = set(List2) 
    if set1.intersection(set2): 
        return True 
    else: 
        return False        