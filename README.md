# Mini-Project-2 Description:

To run these programs ensure you have installed sklearn, nltk, pandas, csv, and numpy. If you have not already, you will need run nltk.download() to download the relevant corpuses (stopwords, sentiment, vader lexicon).

Details of Preprocessing.py:
This program will be called in the Naive.py program. It is not necessary to run it on its own to generate any of our results. 

Details of Naive.py: 
This program is our Bernoulli Naive Bayes Classifier.
JACOB

Details of Multinomial_Scikit.py:
The code can replicate all of our reported results for the Multinomial Naive Bayes classifier.

In order to replicate the results, we have placed comments throughout the code which indicate which lines to "comment-out" in order to generate different feature extraction techniques. Running the code will give both a 5-fold cross validation score, as well as create a csv file containing the predictions of the model for the test data. This csv file can be found in the same folder as the file in which you have saved Multinomial_Scikit.py

One should note that this file must be stored in a folder which contains the csv files 'reddit_train.csv' and 'reddit_test.csv', which pertain to the training and testing data respectively. 

Details of Linear_Scikit.py:
The code can replicate all of our reported results for the Logistic Regression and LinearSVC classifiers. In order to replicate the results, we have placed comments throughout the code which indicate which lines to "comment-out" in order to generate different feature extraction techniques, and which lines to "un-comment" in order to choose a model to form predictions with. 

Since we only ran predictions with Scikit Learn on the test data using the Multinomial Naive Bayes classifier, instructions to generate predictions for the test data using these methods are not included in this file. 


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
