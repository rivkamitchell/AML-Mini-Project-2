# Mini-Project-2 Description:

The report for this project can be found: 
https://drive.google.com/file/d/1HFoHiNzdR7Lb_pkgr5OyqktIZNk_YQLr/view?usp=sharing

To run these programs ensure you have installed sklearn, nltk, pandas, csv, and numpy. If you have not already, you will need run nltk.download() to download the relevant corpuses (stopwords, sentiment, vader lexicon).

Details of preprocessing.py:
This program will be called in the Naive.py program. It is not necessary to run it on its own to generate any of our results. 

Details of naivebayes.py: 
This program is our Bernoulli Naive Bayes Classifier. To test this, refer to Details of Test.py right below.

Details of test.py:
By running this file on terminal, a Bernoulli Naive Bayes model will be initialized, and will train on 2000 comments (for time efficiency), it will keep the top 70000 tfidf scoring features. Then the accuracy score on the training set and a 5-fold validation will be printed in this order. You may modify the 2000 and 70000 parameters in line 22. 


Details of Multinomial_Scikit.py:
The code can replicate all of our reported results for the Multinomial Naive Bayes classifier.

In order to replicate the results, we have placed comments throughout the code which indicate which lines to "comment-out" in order to generate different feature extraction techniques. Running the code will give both a 5-fold cross validation score, as well as create a csv file containing the predictions of the model for the test data. This csv file can be found in the same folder as the file in which you have saved Multinomial_Scikit.py

One should note that this file must be stored in a folder which contains the csv files 'reddit_train.csv' and 'reddit_test.csv', which pertain to the training and testing data respectively. 

Details of Linear_Scikit.py:
The code can replicate all of our reported results for the Logistic Regression and LinearSVC classifiers. In order to replicate the results, we have placed comments throughout the code which indicate which lines to "comment-out" in order to generate different feature extraction techniques, and which lines to "un-comment" in order to choose a model to form predictions with. 

Since we only ran predictions with Scikit Learn on the test data using the Multinomial Naive Bayes classifier, instructions to generate predictions for the test data using these methods are not included in this file. 




