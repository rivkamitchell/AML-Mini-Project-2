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

train = prune(train_data)
# test = prune(test_data)

# Get term-document matrix (Tf-idf weighted document-term matrix)
# The matrix has rows of the form (A,B) C where A = Comment number, B = term index (column this term is a header of), C = Tfidf score for term B in comment A
corpus = train[:,0]
X = vectorizer.fit_transform(corpus)

# Get the terms we have considered in Tf-idf
# terms = vectorizer.get_feature_names()







