# Mini-Project-2

Details of Preprocessing.py:

This file will store the data from the test and training datasets in an array. After, it will lemmatize and remove stopwords and "non-valid" words from each of the comments. Note that lemmatization is only done for POS tag, 'v', meaning verbs. Finally, it computes a TF-IDF score matrix for the training data.

Understanding the TF-IDF score matrix: Each row is of the form (A,B) C, where A is the comment number, B is the index we have assigned to the term, C is term B's TF-IDF score in comment number A.

The packages used in this file are: numpy, pandas, nltk, and scikit. Pandas is used for data set importation, nltk is used for text processing, and scikit is used to compute the TFI-IDF score matrix. 
