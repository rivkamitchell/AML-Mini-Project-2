import sklearn
import pandas as pd
import numpy as np
import csv

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

# Read training csv file, remove ID
df = pd.read_csv('reddit_train.csv', sep = ',', header = None)
train_data = df.values[1:,1:]

# Read test csv file, remove ID
df = pd.read_csv('reddit_test.csv')
test_data = df.values[:,1]

# MAX VOTE
def majority_vote(train_data, test_data):
    (X_train, X_test, y_train, y_test) = train_test_split(train_data[:,0], train_data[:,1], random_state = 0)
    
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    Multinomial_Classification = MultinomialNB().fit(X_train_tfidf, y_train)
    Bernoulli_Classification = BernoulliNB().fit(X_train_tfidf, y_train)
    Decision_Trees_Classification = DecisionTreeClassifier(random_state = 0).fit(X_train_tfidf, y_train)
    Logistic_Regression_Classification = LogisticRegression(random_state = 0, solver = 'lbfgs', multi_class = 'multinomial').fit(X_train_tfidf, y_train)
    

    mc = Multinomial_Classification.predict(count_vect.transform(test_data))
    bc = Bernoulli_Classification.predict(count_vect.transform(test_data))
    dt = Decision_Trees_Classification.predict(count_vect.transform(test_data))
    lr = Logistic_Regression_Classification.predict(count_vect.transform(test_data))
    
    votes = np.array([mc, bc, dt, lr]).T

    predictions = []
    for i in range(0, len(test_data)):
        vote_0 = votes[i][0]
        vote_1 = votes[i][1]
        vote_2 = votes[i][2]
        vote_3 = votes[i][3]
    
        col_votes = [vote_0, vote_1, vote_2, vote_3]

        count_0 = np.count_nonzero(votes[i] == vote_0)
        count_1 = np.count_nonzero(votes[i] == vote_1)
        count_2 = np.count_nonzero(votes[i] == vote_2)
        count_3 = np.count_nonzero(votes[i] == vote_3)

        counts = [count_0, count_1, count_2, count_3]
        
        maj_vote = col_votes[0]
        maj_count = counts[0]

        for j in range(0, len(counts)-1):
            if counts[j] >= maj_count:
                maj_count = counts[j]
                maj_vote = col_votes[j]
        
        predictions += [maj_vote]
    return predictions

def predict(train_data, test_data, model):
    (X_train, X_test, y_train, y_test) = train_test_split(train_data[:,0], train_data[:,1], random_state = 0)

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    if model == 'Bernoulli':
        Bernoulli_Classification = BernoulliNB().fit(X_train_tfidf, y_train)
        return Bernoulli_Classification.predict(count_vect.transform(test_data))
        
    if model == 'Multinomial':
        Multinomial_Classification = MultinomialNB().fit(X_train_tfidf, y_train)
        return Multinomial_Classification.predict(count_vect.transform(test_data))
    
    if model == 'DecisionTree':
        Decision_Trees_Classification = DecisionTreeClassifier(random_state = 0).fit(X_train_tfidf, y_train)
        return Decision_Trees_Classification.predict(count_vect.transform(test_data))
    
    if model == 'Logistic':
        Logistic_Regression_Classification = LogisticRegression(random_state = 0, solver = 'lbfgs', multi_class = 'multinomial').fit(X_train_tfidf, y_train)
        return Logistic_Regression_Classification.predict(count_vect.transform(test_data))

    else: 
        return majority_vote(train_data, test_data)


multinomial_predictions = predict(train_data, test_data, 'Multinomial')
bernoulli_predictions = predict(train_data, test_data, 'Bernoulli')
decision_predictions = predict(train_data, test_data, 'DecisionTree')
logistic_predictions = predict(train_data, test_data, 'Logistic')
majority_predictions = predict(train_data, test_data, 'Majority')
#with open('MultinomialResults.csv', mode = 'w') as file1:
#    writer1 = csv.writer(file1, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
#    writer1.writerow(['id','Category'])
#    for i in range(0, len(multinomial_predictions)):
#        writer1.writerow([str(i), str(multinomial_predictions[i])])

#with open('BernoulliResults.csv', mode = 'w') as file2:
#    writer2 = csv.writer(file2, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
#    writer2.writerow(['id', 'Category'])
#    for i in range(0, len(bernoulli_predictions)):
#        writer2.writerow([str(i), str(bernoulli_predictions[i])])

#with open('DecisionResults.csv', mode = 'w') as file3:
#    writer3 = csv.writer(file3, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
#    writer3.writerow(['id', 'Category'])
#    for i in range(0, len(decision_predictions)):
#        writer3.writerow([str(i), str(decision_predictions[i])])

#with open('LogisticResults.csv', mode = 'w') as file4:
#    writer4 = csv.writer(file4, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
#    writer4.writerow(['id', 'Category'])
#    for i in range(0, len(logistic_predictions)):
#        writer4.writerow([str(i), str(logistic_predictions[i])])

with open('MaxResults.csv', mode = 'w') as file5:
    writer5 = csv.writer(file5, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
    writer5.writerow(['id', 'Category'])
    for i in range(0, len(majority_predictions)):
        writer5.writerow([str(i), str(majority_predictions[i])])