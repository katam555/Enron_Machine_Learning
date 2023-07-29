#!/usr/bin/python3

import joblib
import numpy
import pickle
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
# words_file = "../feature_selection/word_data.pkl" 
# authors_file = "../feature_selection/email_authors.pkl"
# # word_data = joblib.load( open(words_file, "r"))
# # authors = joblib.load( open(authors_file, "r") )
# word_data = pickle.load( open(words_file, "r"))
# authors = pickle.load( open(authors_file, "r") )


words_file = "../feature_selection/word_data_overfit.pkl" 
authors_file = "../feature_selection/email_authors_overfit.pkl"

with open(words_file, "rb") as file:
    word_data = pickle.load(file)

with open(authors_file, "rb") as file:
    authors = pickle.load(file)



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
print(len(features_train))
labels_train   = labels_train[:150]



### your code goes here
from sklearn import tree
from sklearn.metrics import accuracy_score
clf=tree.DecisionTreeClassifier()
clf.fit(features_train,labels_train)
pred=clf.predict(features_test)
acc=accuracy_score(labels_test,pred)
print(acc)
feature_importances_array = clf.feature_importances_
feature_mapping_array = vectorizer.get_feature_names_out()
for i,val in enumerate(feature_importances_array):
    if val>0.2:
        print(val,i,feature_mapping_array[i])

print('word number 33614 with highest importance: {}'.format(feature_mapping_array[33614]))
print('word number 14343 with new highest importance: {}'.format(feature_mapping_array[14343]))
print('word number 14343 with new highest importance: {}'.format(feature_mapping_array[21323]))



