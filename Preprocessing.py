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
df = pd.read_csv('reddit_test.csv')
test_data = df.values[:,1]

# Lemmatize the comment and remove stopwords, and words that are not valid - careful here this gets rid of numbers so we might want to reconsider 
def prune(data):
	for item in data:
		tokens = tokenizer.tokenize(item)
		tokens = [w.lower() for w in tokens]
		filtered = [lemmatizer.lemmatize(w, 'v') for w in tokens if w in valid_words]
		item = ' '.join([w for w in filtered if not w in stop_words and len(w) > 1])
	return data 

def get_classes(data):
    return np.unique(data[:,1])

def get_features(data):
	terms = np.unique(tokenizer.tokenize(' '.join(data)))
	return terms

# Create a matrix where each row pertains to a comment in our data. 
# The last entry of each row contains the subreddit name
# Otherwise, for an entry (i,j) we have that matrix[i][j] = 1 if word j is in comment i, otherwise matrix[i][j] = 0

# Takes as input comments 
def matrix(data, terms):
	matrix = []
	length = len(terms)
	
	for item in data:
		
		comment = tokenizer.tokenize(item)	
		indicator = [0]*length
		count = 0
		for term in terms:
			for word in comment:
				if word==term:
					indicator[count]=1
			count+=1
		
		matrix += [indicator]

	return matrix
  	
def class_num(results):
	classes = np.unique(results)
	num_classes = np.zeros((len(results), 1))

	for i in range(0, len(results)-1):
		num_classes[i] = np.where(classes == results[i])[0][0]

	return (num_classes, classes)

def append_classes(matrix, results):
	return np.concatenate((matrix, results), axis = 1)
	
# Remove low tfidf scores from training data
def remove_tfidf(data, top):
	tfidf_scores = vectorizer.fit_transform(data)
	
	feature_names = vectorizer.get_feature_names()
	
	indices = np.argsort(vectorizer.idf_)[::-1]
	
	top_features = [feature_names[i] for i in indices[:top]]

	for i in range(0, len(data)-1):
		comment = tokenizer.tokenize(data[i])
		filtered = [w for w in comment if w in top_features]
		data[i] = ' '.join(filtered)
	
	return (data, top_features)
	
def get_train_data(data, threshold):
	comments = data[:,0]
	subreddits = data[:,1]

	train_prune = prune(comments)
	(train_tfidf, features) = remove_tfidf(train_prune, threshold)

	train_binary = matrix(train_tfidf, features)

	(results, classes) = class_num(subreddits)
	train_full = append_classes(train_binary, results)

	return (train_full, features, classes)

def get_test_data(data, features):
	test_prune = prune(data)
	test_binary = matrix(test_prune, features)
	return test_binary

(train, features, classes) = get_train_data(train_data[0:100,:], 100)
#print(train_data[:3,0])
#print(make_test_right(test_data)[:3])
#print(test_data)
#print(train_data[:,0])
print(get_test_data(test_data[:100], features))


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

