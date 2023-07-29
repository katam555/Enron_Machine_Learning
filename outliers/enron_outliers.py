#!/usr/bin/python3
import os
import joblib
import sys
import matplotlib.pyplot
import pickle
sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit


# ### read in data dictionary, convert to numpy array
# data_dict = joblib.load( open("../final_project/final_project_dataset.pkl", "rb") )



### your code below
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "rb") )
max1=0
l=[]
for k,v in data_dict.items():
	if v['bonus']=='NaN':
		continue
	else:
	    max1=max(max1,int(v['bonus']))
print(max1)
for k,v in data_dict.items():
	if v['bonus']=='NaN':
		continue
	elif int(v['bonus'])>=50000000:
		print(k)

features = ["salary", "bonus"]
data_dict.pop('TOTAL', 0)
data = featureFormat(data_dict, features)
for k,v in data_dict.items():
	if v['bonus']=='NaN':
		continue
	elif int(v['bonus'])>=5000000 and v['salary']>=1000000:
		print(k)


# ### your code below
# for point in data:
# 	salary = point[0]
# 	bonus = point[1]
# 	matplotlib.pyplot.scatter(salary, bonus)

# matplotlib.pyplot.xlabel('salary')
# matplotlib.pyplot.ylabel('bonus')
# matplotlib.pyplot.show()



