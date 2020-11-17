import numpy as np 
import pandas as pd 
import pickle
import boto3
from collections import defaultdict


batches = ['20200116', '20200217', '20201022']
def data_reader(df_name, batch = batches[-1]):
    '''
    df_name: 'conversations', 'conversations_tag', or 'messages'
    '''
    if batch not in batches:
        batches.append(batch)
    
    bucket = "[redacted_bucket_name]"
    s3 = boto3.resource('s3')
    conversations_df = []
    for b in batches:
        df = pickle.loads(s3.Bucket(bucket).Object('redacted_data/' + df_name + '_' + b + '.pkl').get()['Body'].read())
        conversations_df.append(df)
    return pd.concat(conversations_df)

class tables():
    # Read all tables and make callable as features of tables object.
    def __init__(self):
        self.conv_df = data_reader('conversations', '20201022')
        self.conv_tag_df = data_reader('conversations_tag', '20201022')
        self.messages_df = data_reader('messages', '20201022')
        bucket = "[redacted_bucket_name]"
        s3 = boto3.resource('s3')
        self.tag_dict_df = pickle.loads(s3.Bucket(bucket).Object('tags_dict.pkl').get()['Body'].read())

db = tables()

class tagset():
    # Extract from the tags dictionary table the set of tags under each category of interest.
    def __init__(self):
        self.category = db.tag_dict_df.category.unique()
        self.emotion = db.tag_dict_df[db.tag_dict_df.category == 'Emotion']['full_tag_name'].values
        self.people = db.tag_dict_df[db.tag_dict_df.category == 'People']['full_tag_name'].values
        self.issue = db.tag_dict_df[db.tag_dict_df.category == 'Issue']['full_tag_name'].values
        self.factor = db.tag_dict_df[db.tag_dict_df.category == 'Factor']['full_tag_name'].values
        self.industry = db.tag_dict_df[db.tag_dict_df.category == 'Industry']['full_tag_name'].values
        self.texter = db.tag_dict_df[db.tag_dict_df.category == 'Texter']['full_tag_name'].values

    def by_cat(self, cat):
        if cat == 'Emotion':
            return self.emotion
        if cat == 'People':
            return self.people
        if cat == 'Issue':
            return self.issue
        if cat == 'Factor':
            return self.factor
        if cat == 'Industry':
            return self.industry
        if cat == 'Texter':
            return self.texter
        
tags = tagset() 

class conversation():
    # Extract and make callable features of conversation
    def __init__(self, conv_id):
        self.conv_id = conv_id
        self.extract = self.getExtract()
        self.completion_status = db.conv_df[db.conv_df.conv_id == conv_id]['completion_status'].values[0]
        self.start_time = stringToTimestamp(db.conv_df[db.conv_df.conv_id == conv_id]['start_time'].values[0])
        self.end_time = stringToTimestamp(db.conv_df[db.conv_df.conv_id == conv_id]['end_time'].values[0])
        self.time_span = (self.end_time - self.start_time) / pd.Timedelta('1 hour') # get time span in hours
        self.num_uniq_messages = db.conv_df[db.conv_df.conv_id == conv_id]['num_uniq_messages'].values[0]
        self.num_in = db.conv_df[db.conv_df.conv_id == conv_id]['num_in'].values[0]
        self.num_out = db.conv_df[db.conv_df.conv_id == conv_id]['num_out'].values[0]
        self.emotion_tags = self.getTags('emotion')
        self.issue_tags = self.getTags('issue')
        self.people_tags = self.getTags('people')
        self.inbound_word_count = self.inbound_wc()
        self.outbound_word_count = self.outbound_wc()
        
                
    def getExtract(self):
        # Read seperated messages from the messages table and combined them by conversation id.
        msglist = db.messages_df[db.messages_df.conversation_id == self.conv_id]['redacted_extract']
        dirlist = db.messages_df[db.messages_df.conversation_id == self.conv_id]['direction']
        msg = ''
        for m, d in zip(msglist, dirlist):
            msg += d + ': ' + m + '\n'
        return msg
    
    def getTags(self, tag_category):
        # Extract the tags of interest (in the emotion, issue, and people categories) attached to a converstion 
        # Each category is represented as a dict, with the complete set of tags under this category with default value 0; 
        # if a specific tag was attached to the conversation, update its value to 1
        emotion = {}
        issue = {}
        people = {}
        emotion = defaultdict(lambda: 0, emotion)
        for e in tags.emotion:
            emotion[e]
        issue = defaultdict(lambda: 0, issue)
        for i in tags.issue:
            issue[i]
        people = defaultdict(lambda: 0, people)
        for p in tags.people:
            people[p]        
        conv_tags = db.conv_tag_df[db.conv_tag_df.conv_id == self.conv_id]['tag'].values
        for t in conv_tags:
            if t in tags.emotion:
                emotion[t] = 1
            if t in tags.issue:
                issue[t] = 1
            if t in tags.people:
                people[t] = 1
        if tag_category == 'emotion':
            return emotion
        if tag_category == 'issue':
            return issue
        if tag_category == 'people':
            return people
        
    def printExtract(self):
        print(self.extract)
        
    def hasTag(self, full_tag_name):
        # Return a boolean value whether a given conversation has a tag)
        tt = allTags().byConvid(self.conv_id)        
        if full_tag_name in tt:
            return True
        else:
            return False
        
    def inbound_wc(self):
        inbound_msglist = db.messages_df[(db.messages_df.conversation_id == self.conv_id) & (db.messages_df.direction == 'Inbound')]['redacted_extract'].values
        wc = 0
        for i in inbound_msglist:
            wc += len(i)
        return wc
    
    def outbound_wc(self):
        outbound_msglist = db.messages_df[(db.messages_df.conversation_id == self.conv_id) & (db.messages_df.direction == 'Outbound')]['redacted_extract'].values
        wc = 0
        for i in outbound_msglist:
            wc += len(i)
        return wc
        
class allTags():
    # Extract all tags in the entire dataset from converstion_tags table (in order to see the frequency of each tag)
    def __init__(self):
        self.all = db.conv_tag_df['tag'].values
        
    def byConvid(self, conv_id):
        # Get all tags attached to a given conversation id.
        return db.conv_tag_df[db.conv_tag_df.conv_id == conv_id]['tag'].values
    
    def byCategory(self, category):
        # Get all tags under a given category.
        all_emotion_tags = []
        for e in self.all:
            if e in tags.emotion:
                all_emotion_tags.append(e)
        all_people_tags = []
        for p in self.all:
            if p in tags.people:
                all_people_tags.append(p)
        all_issue_tags = []
        for i in self.all:
            if i in tags.issue:
                all_issue_tags.append(i)
        if category == 'emotion':
            return all_emotion_tags
        if category == 'people':
            return all_people_tags
        if category == 'issue':
            return all_issue_tags
        
class all_ids():
    def __init__(self):
        self.db = tables()
        self.get = self.db.conv_df['conv_id'].values
        self.incomplete = self.db.conv_tag_df[self.db.conv_tag_df.tag == '1.2 Incomplete']['conv_id'].values
        self.initiated_only = self.db.conv_tag_df[self.db.conv_tag_df.tag == '1.3 Incomplete: Initiated only']['conv_id'].values
        self.completed = self.db.conv_tag_df[self.db.conv_tag_df.tag == '1.1 Completed']['conv_id'].values
        self.compl_status_untagged = np.setdiff1d(self.get, np.concatenate((self.initiated_only, self.incomplete, self.completed), axis = None))
        self.great = self.db.conv_tag_df[self.db.conv_tag_df.tag == 'Great Conversation']['conv_id'].values
        self.nongreat = np.setdiff1d(np.concatenate((self.incomplete, self.completed), axis=None), self.great)
        
def stringToTimestamp(timestr):
    # help function to convert time string to pandas timestamp.
    # input: string in the format: '2018-11-16 06:06:08'
    # output: pandas timestamp
    year = int(timestr[0:4])
    month = int(timestr[5:7])
    day = int(timestr[8:10])
    hour = int(timestr[11:13])
    minute = int(timestr[14:16])
    second = int(timestr[17:19])
    timestamp = pd.Timestamp(year, month, day, hour, minute, second)
    return timestamp




