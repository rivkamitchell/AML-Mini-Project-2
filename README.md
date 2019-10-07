# Mini-Project-2 Description:

Details of Preprocessing.py:

This program stores the data from training and test sets. After this, it lemmatizes and removes stopwords and "non-valid" words. By non-valid, we mean strings which contain sequences of letters which do not form words recognized in a spoken language. Finally, a TF-IDF score matrix is computed for the training data.

We can interpret the TF-IDF score matrix as follows: Each row is of the form (A,B) C, where A is the comment number, B is the index we have given to a term, C is the TF-IDF score of term B in comment number A. 

This program uses the packages pandas, nltk, and scikit learn. Pandas is used for text storage, nltk is used to clean the text, and scikit learn is used to compute the TF-IDF matrix for the training set. Before execution, one might have to run nltk.download() in the command line if they have not done so before. 

Details of NAIVE.py:
