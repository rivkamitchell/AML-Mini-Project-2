# Mini-Project-2 Description:

Details of Preprocessing.py:

This program stores the data from training and test sets. After this, it lemmatizes and removes stopwords and "non-valid" words. By non-valid, we mean strings which contain sequences of letters which do not form words recognized in a spoken languages. 

There are several other functions in this file:
1. matrix(data)
This returns an nxm matrix of the form [A|B]. Here A is an nx(m-1) sized matrix where the (i,j)-th entrie is 1 if term j is contained in comment i, and 0 otherwise. B is an nx1 column vector where the i-th entry of B is the subreddit containing comment i. 

2. class_average(data)
This returns an array with entries [A,B] where A is the name of a subreddit and B is the distribution of A in the corpus.

3. feature_average(data,x)
This returns the probability of seeing term x in the text.

4. tfidf(data)
This returns a matrix of TF-IDF scores. We can interpret the TF-IDF score matrix as follows: Each row is of the form (A,B) C, where A is the comment number, B is the index we have given to a term, C is the TF-IDF score of term B in comment number A. 

5. score_function(data,j)
This computes the variable rank of word j in the data.

6. filter_data(data,threshold)
This returns the data with terms which have 'low' variable rankings removed. By low variable ranking we mean that the squared variable ranking of the term is below the given threshold.

The file makes use of the following packages: numpy, pandas, nltk, and scikit learn. In order to run this program, one might need to run nltk.download() and download corpora 'stopwords', 'words', and 'wordnet'.  
