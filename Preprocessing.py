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

def get_classes(data):
    return np.unique(data[:,1])

def get_features(data):
	terms = np.unique(tokenizer.tokenize(' '.join(data[:,0])))
	return terms

# Create a matrix where each row pertains to a comment in our data. 
# The last entry of each row contains the subreddit name
# Otherwise, for an entry (i,j) we have that matrix[i][j] = 1 if word j is in comment i, otherwise matrix[i][j] = 0

# Takes as input comments 
def matrix(data):
	matrix = []
	terms = np.unique(tokenizer.tokenize(' '.join(data)))
	length = len(terms)

	for item in data:
		comment = tokenizer.tokenize(item)	
		indicator = [0]*length
		
		for word in comment: 
			where = np.where(terms == word)[0][0]
			indicator[where] = 1
		
		matrix += [indicator]

	return matrix

# Get the test data
def append_classes(data):
	term_matrix = np.array(matrix(data[:,0]))
	classes = get_classes(data)
	
	class_column = np.zeros((len(data[:,1]),1), dtype = int)	

	for i in range(0, len(data[:,1])-1):
		class_column[i] = np.where(classes == data[i][1])[0][0]

	train_data = np.concatenate((term_matrix, class_column), axis = 1)
	return train_data

# Remove low tfidf scores from training data
def remove_tfidf(data, top):
	tfidf_scores = vectorizer.fit_transform(data[:,0])
	feature_names = vectorizer.get_feature_names()
	indices = np.argsort(vectorizer.idf_)[::-1]
	
	top_features = [feature_names[i] for i in indices[:top]]

	for i in range(0, len(data[:,1])-1):
		comment = tokenizer.tokenize(data[i][0])
		filtered = [w for w in comment if w in top_features]
		data[i][0] = ' '.join(filtered)
	
	return data
	
def get_train_data(data, threshold):
	train_prune = prune(data)
	train_tfidf = remove_tfidf(train_prune, threshold)
	train_binary = append_classes(train_tfidf)

	return train_binary

def get_test_data(data):
	test_prune = prune(data)
	test_binary = matrix(test_prune)

	return test_binary


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

################### HOW TO TEST THE FUNCTIONS ###################################

#print(get_train_data(train_data[0:100,:], 100))

# TEST PRUNE FUNCTION
#train_pruned = prune(train_data)
#print(train_pruned)

# TEST MATRIX FUNCTION
#train_pruned = prune(train_data)
#train = matrix(train_pruned[0:100,:])
#print(train)

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

    		






