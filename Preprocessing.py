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
		item[0] = ' '.join([w for w in filtered if not w in stop_words and len(w) > 1])
	return data 

# Get term-document matrix (Tf-idf weighted document-term matrix)
# The matrix has rows of the form (A,B) C where A = Comment number, B = term index (column this term is a header of), C = Tfidf score for term B in comment A
def tfidf(data):
    return vectorizer.fit_transform(data[:,0])

# TRAINING FUNCTIONS

# Convert strings into numerical values
def change_to_numeric(data):
	classes = np.unique(data[:,1])
	features = np.unique(' '.join(data[:,0]))

	for comment in data:
		comment[1] = classes.index(comment[1])
		for item in comment[0].split():
			item = features.index(item)
	
	return data

# Returns distributions of the classes as an array with entries [class, class_distribution]
def class_average(data):
	number_comments = len(data[:,1])
	classes = np.unique(data[:,1])
	
	means = []
	for x in classes: 
		mean = np.count_nonzero(data[:,1] == x)/number_comments
		means += [mean]	

	return means 

# Returns probability of seeing term x in a comment
def feature_average(data, x):
	text = tokenizer.tokenize(' '.join(data[:,0]))
	mean = np.count_nonzero(text == x)/len(text)
	return mean

# Variable Ranking
def score_function(data,j):
	class_averages = class_average(data)
	
	y_square = 0
	x_square = 0
	y_linear = 0
	x_linear = 0

	numerator = 0
	for i in range(0, len(data[:,1]) - 1):
		y_linear = (data[i][1] - class_averages[data[i][1]])

		y_square += (data[i][1] - class_averages[data[i][1]])*(data[i][1] - class_averages[data[i][1]])
		
		if j in data[i][0]: 
			x_linear = (1 - feature_average(data, j))
			numerator += x_linear*y_linear

			x_square += (1 - feature_average(data, j))*(1 - feature_average(data,j))
		else: 
			x_linear = -feature_average(data,j)
			numerator += x_linear*y_linear
			x_square += (feature_average(data, j))*(feature_average(data,j))

	denominator = x_square*y_square

	return numerator/denominator

# Get rid of low ranking terms
def filter_data(data, threshold):
	terms = np.unique((' '.join(data[:,0])).split())

	keep_terms = []
	for j in terms: 
		if (score_function(data, j))*(score_function(data,j)) >= threshold: keep_terms.insert(j,0)

	for comment in data:
		filtered = [w for w in comment[0] if w in keep_terms]
		comment[0] = ' '.join(filtered)


    		






