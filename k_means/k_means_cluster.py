#!/usr/bin/python3

""" 
    Skeleton code for k-means clustering mini-project.
"""

import os
import joblib
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit

def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()



### load in the dict of dicts containing all the data on each person in the dataset
data_dict = joblib.load( open("../final_project/final_project_dataset.pkl", "rb") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)

max1=float('-inf')
min1=float('inf')

for k,v in data_dict.items():
    if v['exercised_stock_options']=='Nan':
        continue
    else:
        max1=max(max1,float(v['exercised_stock_options']))
        min1=min(min1,float(v['exercised_stock_options']))
print(max1,min1)

max2=float('-inf')
min2=float('inf')

for k,v in data_dict.items():
    if v['salary']=='Nan':
        continue
    else:
        max2=max(max2,float(v['salary']))
        min2=min(min2,float(v['salary']))
print(max2,min2)



### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
# feature_3 = "total_payments"
poi  = "poi"
features_list = [poi, feature_1, feature_2]
data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )


### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)
for f1, f2 in finance_features:
    plt.scatter( f1, f2)
plt.show()

### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=2)
kmeans.fit(finance_features)
pred=kmeans.predict(finance_features)

from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
finance_features = min_max_scaler.fit_transform(finance_features)
finance_features_test = min_max_scaler.transform([[200000., 1000000.]])
print('finance_features transformed with min_max_scaler: {}'.format(finance_features_test))




### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print("No predictions object named pred found, no clusters to plot")
