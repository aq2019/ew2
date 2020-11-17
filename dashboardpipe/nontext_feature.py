from load_data import tables, conversation, stringToTimestamp, all_ids, allTags

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gensim
from gensim import corpora
from textblob import TextBlob





db = tables()

class feature():
    def __init__(self, conv_id):
        self.id = conv_id
        self.all_info = db.messages_df[db.messages_df.conversation_id == self.id].copy(deep=True) #slice of dataframe that includes all information of the conversation
        self.messages = self.all_info['redacted_extract'].values
        self.last_few = 5 # define how many messages at the end of the conversation are considered as last few
        self.in_messages = self.all_info[self.all_info['direction'] == 'Inbound']['redacted_extract'].values
        self.num_in_messages = len(self.in_messages)
        if self.num_in_messages > self.last_few:
            self.last_in_msg = self.in_messages[-self.last_few:]
        else:
            self.last_in_msg = self.in_messages
        self.out_messages = self.all_info[self.all_info['direction'] == 'Outbound']['redacted_extract'].values
        self.num_out_messages = len(self.out_messages)
        if self.num_out_messages > self.last_few:
            self.last_out_msg = self.out_messages[-self.last_few:]
        else:
            self.last_out_msg = self.out_messages
        self.direction = self.all_info['direction'].values
        self.timestamp = self.all_info['message_date'].values
        #self.timestamp = None
        #self.timeStamp()
        self.start_time = pd.to_datetime(db.conv_df[db.conv_df.conv_id == self.id]['start_time'].values[0])
        self.end_time = pd.to_datetime(db.conv_df[db.conv_df.conv_id == self.id]['end_time'].values[0])
        
        
    
    def timeindex(self):
        ts = pd.to_datetime(self.all_info['message_date'].values)
        #time_stamp = np.asarray([stringToTimestamp(t) for t in ts])
        return ts
    
    def timeStamp(self):
        ts = self.all_info['message_date'].values
        self.timestamp = np.asarray([stringToTimestamp(t) for t in ts])
        
    
    def inbound_timeindex(self):
        inbound = self.all_info[self.all_info['direction'] == 'Inbound']
        tidx = pd.to_datetime(inbound['message_date'].values)
        return tidx
    
    def inbound_timestamp(self):
        inbound = self.all_info[self.all_info['direction'] == 'Inbound']
        its = inbound['message_date'].values
        inbound_timestamp = np.asarray([stringToTimestamp(t) for t in its])
        return inbound_timestamp
    
    def outbound_timeindex(self):
        outbound = self.all_info[self.all_info['direction'] == 'Outbound']
        tidx = pd.to_datetime(outbound['message_date'].values)
        return tidx
    
    def outbound_timestamp(self):
        outbound = self.all_info[self.all_info['direction'] == 'Outbound']
        its = outbound['message_date'].values
        outbound_timestamp = np.asarray([stringToTimestamp(t) for t in its])
        return outbound_timestamp
    
    def response_rate(self):
        time_delta = [0]
        for i in range(len(self.direction)-1):
            time_delta.append((self.timestamp[i+1] - self.timestamp[i])/np.timedelta64(1,'s'))
        
        inbound_delta = []
        outbound_delta = []
        for d in range(len(self.direction)-1):
            if self.direction[d+1] != self.direction[d]:
                if self.direction[d+1] == 'Outbound':
                    if time_delta[d+1] < 3600 and time_delta[d+1] > 0: # time delta larger than 1 hour will be ignored; time delta less than 0 are due to data exportation bug and will be ignored
                        outbound_delta.append(time_delta[d+1])
                if self.direction[d+1] == 'Inbound':
                    if time_delta[d+1] < 3600 and time_delta[d+1] > 0:
                        inbound_delta.append(time_delta[d+1])
        if len(inbound_delta) == 0:
            inbound_rate = 3600
        else:
            inbound_delta = np.asarray(inbound_delta)        
            inbound_rate = int(np.average(inbound_delta))
        if len(outbound_delta) == 0:
            outbound_rate = 3600
        else:
            outbound_delta = np.asarray(outbound_delta)
            outbound_rate = int(np.average(outbound_delta))
        return inbound_rate, outbound_rate
        
    def message_volume(self):
        in_vol = 0
        out_vol = 0
        for i in range(len(self.direction)):
            if self.direction[i] == 'Inbound':
                in_vol += len(self.messages[i].split())
            else:
                out_vol += len(self.messages[i].split())
        if in_vol != 0 and out_vol != 0:       
            ratio = in_vol/out_vol
        else:
            ratio = 0
        return in_vol, out_vol, ratio
    
    def in_dict(self):
        # construct a dictionary for inbound messages
        texts = [[word for word in word_tokenize(message) if word.isalpha()] for message in self.in_messages]
        msg_dict = corpora.Dictionary(texts)
        return msg_dict

    def out_dict(self):
        # construct a dictionary for outbound messages
        texts = [[word for word in word_tokenize(message) if word.isalpha()] for message in self.out_messages]
        msg_dict = corpora.Dictionary(texts)
        return msg_dict
    
    def last_in_dict(self):
        # construct a dictionary for the last few inbound messages
        texts = [[word for word in word_tokenize(message) if word.isalpha()] for message in self.last_in_msg]
        msg_dict = corpora.Dictionary(texts)
        return msg_dict
    
    def last_out_dict(self):
        # construct a dictionary for the last few outbound messages
        texts = [[word for word in word_tokenize(message) if word.isalpha()] for message in self.last_out_msg]
        msg_dict = corpora.Dictionary(texts)
        return msg_dict
    
    def feedback(self):
        # check if the last few outbound messages contain 'feedback'
        feedback = {'feedback'}
        for m in self.last_out_msg:
            for x in feedback:
                if x in m:
                    return 1
                else:
                    continue 
        return 0
    
    def checkin(self):
        # check if the last few outbound messages contain 'check in', 'checking in' or similar phrases
        checkin = {'check in', 'checking in', 'check-in'}
        for m in self.last_out_msg:
            for x in checkin:
                if x in m:
                    return 1
                else:
                    continue 
        return 0
    
    def check_continue(self):
        # check if the last few outbound messages contain 'continue' or similar phrases
        check_continue = {'like to continue', 'want to continue'}
        for m in self.last_out_msg:
            for x in check_continue:
                if x in m:
                    return 1
                else:
                    continue 
        return 0
    

    def come_back(self):
        # check if the last few inbound messages contain 'comeback', 'check back', 'text back' or similar phrases
        comeback = {'come back', 'check back', 'text back', 'try back', 'return', 'reach back', 'another time', 'later'}
        for m in self.last_in_msg:
            for x in comeback:
                if x in m:
                    return 1
                else:
                    continue 
        return 0

    
    def thank(self):
        # check if the last few inbound messages contain phrases of appreciation
        thank = {'thanks', 'thank', 'appreciate', 'helpful', 'thx'}
        for m in self.last_in_msg:
            for x in thank:
                if x in m:
                    return 1
                else:
                    continue 
        return 0
    
    def pop_up(self):
        # check if the last few outbound messages contain phrases like 'things pop up'
        pop_up = {'pop up', 'pulled away', 'pull people away', 'slow down'}
        for m in self.last_out_msg:
            for x in pop_up:
                if x in m:
                    return 1
                else:
                    continue
        return 0
    
    def conversation_last(self):
        # check if the outbound messages contain phrases like 'our conversations last about an hour'
        conversation_last = {'conversations last', 'conversations take', 'back and forth'}
        for m in self.out_messages:
            for x in conversation_last:
                if x in m:
                    return 1
                else:
                    continue
        return 0
   

    #def hi_this_is(self):
        # check how many 'Hi, this is' appear in outbound message
    #    hi_this_is = {'Hi, this is', 'hi, this is'}
    #    ct = 0
    #    for m in self.out_messages:
    #        for x in hi_this_is:
    #            if x in m:
    #                ct += 1
    #            else:
    #                continue
    #    return ct
    
    def our_best_to_you(self):
        # check if outbound messages include phrases like 'All our best to you'
        our_best = {'our best to you', 'all the best'}
        for m in self.out_messages:
            for x in our_best:
                if x in m:
                    return 1
                else:
                    continue 
        return 0
    
    
    def resources(self):
        # check if outbound messages include 'resources' (i.e. resources have been offered)
        resources = {'resources'}
        for m in self.out_messages:
            for x in resources:
                if x in m:
                    return 1
                else:
                    continue 
        return 0
    
    def close_this(self):
        # check if the last few outbound messages include 'close this for now'
        close = {'close this for now'}
        for m in self.last_out_msg:
            for x in close:
                if x in m:
                    return 1
                else:
                    continue
        return 0
    
    def sentiment_score(self):
        in_sentiment = []
        out_sentiment = []
        for m in self.in_messages:
            sen = TextBlob(m)
            in_sentiment.append(sen.sentiment.polarity)
        for m in self.out_messages:
            sen = TextBlob(m)
            out_sentiment.append(sen.sentiment.polarity)
        in_sentiment = np.asarray(in_sentiment)
        out_sentiment = np.asarray(out_sentiment)
        return in_sentiment, out_sentiment    
        
        
    

        
    
    
    
class classifier():
    # a rule-based classifier for completion status
    def __init__(self, conv_id):
        self.id = conv_id
        
    def initiated_only(self):
        f = feature(self.id)
        if len(f.direction()) < 15:
            return True
        else:
            return False
        
    def completion(self):
        f = feature(self.id)        
        if len(f.messages()) >= 4:
            last_msg_latency = (f.timestamp()[-1] - f.timestamp()[-2]).days

            if last_msg_latency >= 1:
                if 'feedback' in f.messages()[-2]:
                    last_3_out = sum(f.direction()[-5:-2] == np.array(['Outbound', 'Outbound', 'Outbound']))
                    if last_3_out == 3:
                        return False
                    else:
                        return True
                else:
                    last_3_out = sum(f.direction()[-4:-1] == np.array(['Outbound', 'Outbound', 'Outbound']))
                    if last_3_out == 3:
                        return False
                    else:
                        return True
            if last_msg_latency < 1:
                if 'feedback' in f.messages()[-1]:
                    last_3_out = sum(f.direction()[-4:-1] == np.array(['Outbound', 'Outbound', 'Outbound']))
                    if last_3_out == 3:
                        return False
                    else:
                        return True
                else:
                    last_3_out = sum(f.direction()[-3:] == np.array(['Outbound', 'Outbound', 'Outbound']))
                    if last_3_out == 3:
                        return False
                    else:
                        return True
        else:
            return False
            
        
    
        


