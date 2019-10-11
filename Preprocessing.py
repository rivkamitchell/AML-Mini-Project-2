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

# Create a matrix where each row pertains to a comment in our training data. 
# The last entry of each row contains the subreddit name
# Otherwise, for an entry (i,j) we have that matrix[i][j] = 1 if word j is in comment i, otherwise matrix[i][j] = 0
def matrix(data):
	matrix = []
	classes = get_classes(data)
	
	terms = get_features(data)
	length = len(terms) + 1

	for item in data:
		comment = tokenizer.tokenize(item[0])	
		indicator = [0]*length
		
		for word in comment: 
			where = np.where(terms == word)[0][0]
			indicator[where] = 1
		
		indicator[length-1] = np.where(classes == item[1])[0][0]
		matrix += [indicator]

	return matrix

# We will change this function as we decide which features to include
def get_features(data):
	return np.unique(tokenizer.tokenize(' '.join(data[:,0])))

def get_classes(data):
    return np.unique(data[:,1])

# Returns distributions of the classes as an array with entries [class, class_distribution]
def class_average(data):
	number_comments = len(data[:,1])
	classes = np.unique(data[:,1])
	means = []
	for x in classes: 
		mean = np.count_nonzero(data[:,1] == x)/number_comments
		means += [[x, mean]]	

	return means 

# Returns probability of seeing term x in a comment
def feature_average(data, x):
	text = tokenizer.tokenize(' '.join(data[:,0]))
	mean = text.count(x)/len(text)
	return mean

# Get term-document matrix (Tf-idf weighted document-term matrix)
# The matrix has rows of the form (A,B) C where A = Comment number, B = term index (column this term is a header of), C = Tfidf score for term B in comment A
def tfidf(data):
	return vectorizer.fit_transform(data[:,0])

# TFIDF
def tf_idf(data, term):
	tf = 0
	cf = 1
	for comment in data[0,:]:
		words = tokenizer.tokenize(comment)
		comment_frequency = np.count_nonzero(words == term)
		if comment_frequency != 0:
			tf += comment_frequency
			cf += 1

	idf = np.log(len(data[:,1])/cf)

	return tf*idf

# Variable Ranking
def score_function(data,j):
	class_averages = class_average(data)
	classes = np.unique(data[:,1])
	y_square = 0
	x_square = 0
	y_linear = 0
	x_linear = 0

	numerator = 0
	for i in range(0, len(data[:,1]) - 1):
		y_class = data[i][1]
		where = np.where(classes == y_class)[0][0]
		y_linear = (where - class_averages[where][1])

		y_square += y_linear*y_linear
		
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
def vr_filter(data, threshold):
	terms = np.unique(tokenizer.tokenize(' '.join(data[:,0])))

	keep_terms = []
	for word in terms: 
		if (int(score_function(data, word)))*(int(score_function(data,word))) >= int(threshold): 
			keep_terms += [word]

	for comment in data:
		filtered = [w for w in tokenizer.tokenize(comment[0]) if w in keep_terms]
		comment[0] = ' '.join(filtered)

	return data

def tfidf_filter(data, threshold):
	terms = np.unique(tokenizer.tokenize(' '.join(data[:,0])))
	keep_terms = []
	for word in terms:
		if tf_idf(data, word) >= threshold:
			keep_terms += [word]
	
	for comment in data:
		filtered = [w for w in tokenizer.tokenize(comment[0]) if w in keep_terms]
		comment[0] = ' '.join(filtered)
	
	return data
################################### HOW TO TEST THE FUNCTIONS ###################################

# TEST PRUNE FUNCTION
#train_pruned = prune(train_data)
#print(train_pruned)

# TEST MATRIX FUNCTION
train_pruned = prune(train_data)
train = matrix(train_pruned[0:100,:])
print(train)

# TEST CLASS AVERAGES
#train_pruned = prune(train_data)
#class_averages = class_average(train_pruned)
#print(class_averages)

# TEST FEATURE AVERAGES
#train_pruned = prune(train_data)
#comment = tokenizer.tokenize(train_pruned[0][0])
#feature_avrg = feature_average(train_pruned, comment[0])
#print(feature_avrg)

# TEST TFIDF
#train_pruned = prune(train_data)
#tf_idf = tfidf(train_pruned)
#print(tf_idf)

# TEST SCORE VARIABLE RANKING FUNCTION
#train_pruned = prune(train_data)
#comment = tokenizer.tokenize(train_pruned[0][0])
#word = comment[0]
#score = score_function(train_pruned[0:100,:], word)
#print(score)

# TEST FILTER VARIABLE RANKING FUNCTION
#train_pruned = prune(train_data)
#train_filtered = vf_filter(train_pruned[0:50,:], 0.5)
#print(train_filtered)

# TEST SCORE TFIDF FUNCTION
#train_pruned = prune(train_data)
#comment = tokenizer.tokenize(train_pruned[0][0])
#word = comment[0]
#score = tf_idf(train_pruned[0,100,:], word)
#print(score)

# TEST FILTER TFIDF FUNCTION
#train_pruned = prune(train_data)
#train_filtered = tfidf_filter(train_pruned, 0.5)
#print(train_filtered)

    		






