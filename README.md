# Mini-Project-2

Details of Preprocessing.py

This file will store the data from the test and training datasets in numpy array. After, it will lemmatize and remove stopwords and "non-valid" words from each of the comments. Note that lemmatization is only done for POS tag, 'v', meaning verbs. Finally, it computes a TF-IDF score matrix for the training data.

Understanding the TF-IDF score matrix: Each row is of the form (A,B) C where A is the comment number, B is the index we have assigned to the term, C is term B's TF-IDF score in comment number A

