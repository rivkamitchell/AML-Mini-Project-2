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




