from models import logisticRegr
from collections import defaultdict

from load_data import tables, tagset, conversation, allTags, all_ids
from utilities import readGlove, split_data
from preprocess import text_to_wordlist, word_embedding, doc_embedding, get_data

# train models for all labels
emotion_labels = ['Anxiety/Stress', 'Confidence', 'Stuck', 'Undervalued', 'Fear', 'Frustrated', 'Uncertain', 'Bullied']

issue_labels = ['Benefits or Leave',  'Communication',
       'Discrimination', 'Harassment', 
       'Payroll or time issue', 'Performance',
       'Workload/Hours', 'Job Search']

people_labels = ['Manager', 'Coworker']

categories = [emotion_labels, people_labels, issue_labels]

ids = all_ids().get
'''
def train_one_label(label, ids, model):
    if model == 'LogisticRegression':
        m = logisticRegr(label)
        m.split(ids)
        m.train()
        m.save()

for cat in categories:
    for lb in cat:
        train_one_label(lb, ids, 'LogisticRegression')
        
'''
# make prediction for one data point
txt = 'I have been working at my current job for about 6 months. It has been very high stress the entire time and I have worked a lot of overtime just to keep up (without pay as I am salaried). I have a couple coworkers who are toxic and continue piling excess work on me, making me feel like I am not doing a good job. I have started rejecting their requests and it is causing a LOT of stress between departments. The branch manager is not on my side, and he has made a few rude and degrading comments to me when I bring up these issues. I work in a very small office (6 people). I have walked out a few times to cry, scream, etc.'
nd = [text_to_wordlist(txt)]



pred_one_data = defaultdict()
for cat in categories:
    for lb in cat:
        n = logisticRegr(lb)
        n.load()
        pred_one_data[lb] = n.predict(new_data = nd)[0]

print(pred_one_data)

        


