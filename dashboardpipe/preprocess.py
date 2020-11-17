import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer 
nltk.download('stopwords') 
nltk.download('punkt')
nltk.download('wordnet')
import re
import random
from collections import defaultdict
import gensim
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec, KeyedVectors
from gensim import corpora
from gensim import models
import pickle

from load_data import tables, tagset, conversation, allTags, all_ids
from utilities import readGlove


customized_stopwords = {'redacted', 'Inbound', 'inbound', 'Outbound', 'outbound', '[', ']', 'empower', 'work'}


def get_corpus(ids):
    # input: numpy array: list of conversation ids
    # output: tuple (a list of wordlist with each wordlist representing a document (conversation), ids)
    cor = []
    for i in ids:
        doc = text_to_wordlist(conversation(i).getExtract())
        cor.append(doc)
    return cor, ids

def get_data(ids, label):
    # input - ids of conversation and target label
    # output - texts: list of word lists; y: numpy array of boolean values for the label
    text = []
    y = []
    for i in ids:
        conv = conversation(i)
        wordlist = text_to_wordlist(conv.getExtract())
        y_value = conv.hasTag(label)
        text.append(wordlist)
        y.append(y_value)
    return ids, text, np.asarray(y)

def text_to_wordlist(input_str):
    # Remove punctuations from text, lowercase, remove stop words, tokenize
    tt = re.sub('[^a-zA-z0-9\s]','', input_str) 
    tt = tt.lower() 
    word_tokens = nltk.word_tokenize(tt)    
    stop_words = set(stopwords.words('english')).union(customized_stopwords)
    wordlist = [w for w in word_tokens if not w in stop_words]
    lemmatizer = WordNetLemmatizer() 
    lemmatized_wordlist = [lemmatizer.lemmatize(w) for w in wordlist]
    return lemmatized_wordlist



class word_embedding():
    def __init__(self, method = None, corpus = None, new_doc = None):
        if method == 'GloVe':
            self.w2v = self.GloVe()
        else:
            self.w2v = self.gensim_w2v(corpus)
        
    def GloVe(self):
        return pickle.load(open('glove50d.sav', 'rb'))
    
    def gensim_w2v(self, corpus, new_doc = None):
        # corpus: a list of word lists
        # new_doc: list of wordlist
        # output: word vectors. Usage: get numpy vector of a word: vector = word_vectors['computer']
        # corpus = pickle.load(open(corpus_path, 'rb')) # alternatively, feed in a corpus path
        model = Word2Vec(corpus, size = 100, window=5, min_count=1, workers=4)     
        # if new documents are fed in, continue training the model
        if new_doc:
            model.train(new_doc, total_examples = len(new_doc), epochs = 1)
        word_vectors = model.wv        
        return word_vectors

    
    
class doc_embedding():
    def __init__(self, method = None, word_embedding = None, weight = None, corpus = None):
        self.tfidf_model = None
        self.dictionary = None
        # self.d2v_from_w2v = None
        self.d2v_model = None
        # self.d2v = None
        
    def gensim_train(self, corpus):
        # corpus has to to a list of word lists, with each word list representing a document
        # corpus = pickle.load(open(corpus_path, 'rb')) # alternatively, feed in a corpus path
        tagged_corpus = []
        for i in range(len(corpus)):
            tagged_corpus.append(gensim.models.doc2vec.TaggedDocument(corpus[i], [i]))
        # Instantiate a Doc2Vec model
        model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
        # Build a vocabulary
        model.build_vocab(tagged_corpus)
        vocab = model.wv      
        # Train model
        model.train(tagged_corpus, total_examples=model.corpus_count, epochs=model.epochs)
        self.d2v_model = model
        
    def gensim_infer(self, new_doc):
        # input: new_doc - a list of wordlist, each wordlist representing a new document
        # output: doc_vec - an array of vectors, each vector is a document representation
        doc_vec = []
        for doc in new_doc:
            inferred_vector = self.d2v_model.infer_vector(doc)
            doc_vec.append(inferred_vector)
        return np.asarray(doc_vec)
                
    def w2v_to_d2v(self, w2v, weights):
        # input: w2v - word embedding; weights - list of lists of tuples (word : weight), each list of tuples representing a document. 
        # w2v has to have 'ukn' entry
        # output: document embedding for a collection of documents (type: array)
        dim = 50 # word vector dimension
        doc_embeddings = []
        for one_doc in weights:
            weighted_docvec = np.zeros(dim)
            s = 0
            for (word, w) in one_doc:
                s += w
                if word in w2v.keys():
                    word_vector = w2v[word]
                else:
                    word_vector = w2v['ukn']
                weighted = w*word_vector
                weighted_docvec = np.add(weighted_docvec, weighted)
            normalized_weighted_docvec = weighted_docvec/s
            doc_embeddings.append(normalized_weighted_docvec)
        return np.asarray(doc_embeddings)
                
    def get_tfidf(self, texts, training_texts = None):
        # input: training_texts - a list of wordlist; texts - a list of wordlists
        # output: [[(doc1word1, weight), (doc1word2, weight), ...], [(doc2word1, weight), (doc2word2, weight), ...], ...]
        if training_texts is not None:
            dictionary = corpora.Dictionary(training_texts) # build a dictionary
            self.dictionary = dictionary # save dictionary as a class feature
            corpus = [self.dictionary.doc2bow(text) for text in training_texts] # corpus with bag-of-word vector representation
            tfidf_model = models.TfidfModel(corpus) # initialize a bag of word model
            self.tfidf_model = tfidf_model # keep tfidf model as class feature
        
        tfidf = [] # a list of list of tuples
        for text in texts:
            weights = self.tfidf_model[self.dictionary.doc2bow(text)]
            one_doc = [] # representation of one document: list of tuples (word, weight of the word)
            for ix, w in weights:
                word = self.dictionary[ix] 
                item = (word, w)
                one_doc.append(item)
            tfidf.append(one_doc)                
        return tfidf
        
    # def save_model(self):
        
    
    # def load_model(self):
        
    
      
        
        

