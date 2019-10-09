import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

import sklearn 
from sklearn.feature_extraction.text import TfidfVectorizer

import naivebayes


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

def listify_test(data):
	matrix=[]
	for item in data:
		line=item[0].split()
		matrix+=[line]
	return matrix


def listify(data):
	matrix=[]
	result=[]
	for item in data:
		lign=item[0].split()
		matrix+=[lign]
		result+=item[1].split()
	return (matrix, result)

# Lemmatize the comment and remove stopwords, and words that are not valid - careful here this gets rid of numbers so we might want to reconsider 
#returns a matrix 

def prune(data):
	matrix=[]
	for item in data:
		tokens = [w.lower() for w in item]
		filtered=[lemmatizer.lemmatize(w, 'v') for w in tokens if w in valid_words]
		temp=[w for w in filtered if not w in stop_words and len(w)>1]
		matrix+=[temp]
	return matrix 


def classify(result):
	classes=set(result)
	classes=list(classes)
	return(classes) #list of classes, so they are indexedà

def words_giver(data):
	words=set()
	for lign in data:
		temp=set(lign)
		words=words.union(temp)
	words=list(words)
	return words



# Get term-document matrix (Tf-idf weighted document-term matrix)
# The matrix has rows of the form (A,B) C where A = Comment number, B = term index (column this term is a header of), C = Tfidf score for term B in comment A
def tfidf(data):
    return vectorizer.fit_transform(data[:,0])





def binary_data(data, words, results, classes): #words is a set of words, the size of it is the number of features
	matrix=np.zeros((len(results), len(words)+1), dtype=int)
	lign_count=0
	for lign in data:
		for i in range(len(words)):
			for w in lign:
				if w==words[i]:
					matrix[lign_count][i]=1
					
		lign_count+=1


	result_count=0
	for w in results:
		for i in range(len(classes)):
			if w==classes[i]:
				matrix[result_count][-1]=i
		result_count+=1

	return matrix

def binary_test_data(data, words, classes): #words is a set of words, the size of it is the number of features
	matrix=np.zeros((len(data), len(words)), dtype=int)
	lign_count=0
	for lign in data:
		for i in range(len(words)):
			for w in lign:
				if w==words[i]:
					matrix[lign_count][i]=1
					
		lign_count+=1

	return matrix
