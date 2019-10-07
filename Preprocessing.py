import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

import sklearn 
from sklearn.feature_extraction.text import TfidfVectorizer

tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()
vectorizer = TfidfVectorizer()
stop_words = set(stopwords.words('english'))
valid_words = set(words.words())

# Read training csv file, remove ID
df = pd.read_csv('reddit_train.csv', sep = ',', header = None)
train_data = df.values[1:,1:]

# Read test csv file, remove ID
df = pd.read_csv('reddit_test.csv', sep = ',', header = None)
test_data = df.values[1:,1:]

# Lemmatize the comment and remove stopwords, and words that are not valid - careful here this gets rid of numbers so we might want to reconsider 
def prune(data):
	for item in data:
		tokens = tokenizer.tokenize(item[0])
		tokens = [w.lower() for w in tokens]
		filtered = [lemmatizer.lemmatize(w, 'v') for w in tokens if w in valid_words]
		item[0] = ' '.join([w for w in filtered if not w in stop_words])
	return data 

# Get term-document matrix (Tf-idf weighted document-term matrix)
# The matrix has rows of the form (A,B) C where A = Comment number, B = term index (column this term is a header of), C = Tfidf score for term B in comment A
def tfidf(data):
    return vectorizer.fit_transform(data[:,0])

# TRAINING FUNCTIONS

# Convert class name into numerical value
def assign_class_num(data):
	classes = np.unique(data[:,1])
	for item in data[:,1]:
		item = classes.index(item)
	
	return data

# Returns distributions of the classes as an array with entries [class, class_distribution]
def class_average(data):
	number_comments = len(data[:,1])
	classes = np.unique(data[:,1])
	
	means = []
	for x in classes: 
		mean = np.count_nonzero(data[:,1] == x)/number_comments
		means += [x,mean]	

	return means 

# Returns probability of seeing term x in a comment
def feature_average(data, x):
	text = tokenizer.tokenize(data[:,0].flatten())
	mean = np.count_nonzero(text == x)/len(text)
	return mean

train = prune(train_data)
# test = prune(test_data)






