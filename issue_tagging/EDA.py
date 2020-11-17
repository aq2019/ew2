import matplotlib.pyplot as plt
import pandas as pd
from load_data import tables, tagset, conversation, allTags, all_ids



db = tables()
all_tags = allTags()

completion = db.conv_df['completion_status'].value_counts()
completion.plot(kind = 'bar', width = 0.3, title = 'Frequency of Completion Status')

all_emotion_tags = pd.DataFrame(all_tags.byCategory('emotion'))
emotion_count = all_emotion_tags[0].value_counts()
emotion_count.plot(kind = 'barh', title = 'Frequency of Emotion Classes')

all_issue_tags = pd.DataFrame(all_tags.byCategory('issue'))
issue_count = all_issue_tags[0].value_counts()
issue_count.plot(kind = 'barh', title = 'Frequency of Issue Classes' )

all_people_tags = pd.DataFrame(all_tags.byCategory('people'))
people_count = all_people_tags[0].value_counts()
people_count.plot(kind = 'barh', title = 'Frequency of People Classes')


# How many emation tags, issue tags, people tags does each conversation have?
def num_tag_per_id(tag_category):
    from collections import Counter, defaultdict
    cnt = defaultdict()
    for conv_id in all_ids().get:
        if tag_category == 'emotion':
            tag = conversation(conv_id).emotion_tags
        if tag_category == 'issue':
            tag = conversation(conv_id).issue_tags
        if tag_category == 'people':
            tag = conversation(conv_id).people_tags 
        s = 0
        for k, v in tag.items():
            s += v
        cnt[conv_id] = s
    tf = [v for (k, v) in cnt.items()]
    c = Counter(tf)
    return cnt, c

# Get the list of conversations with a given tag
def getListOfConvsByTag(full_tag_name):
    
    li = []
    for conv_id in all_ids().get:
        if conversation(conv_id).hasTag(full_tag_name):
            li.append(conv_id)
    return li
    

    