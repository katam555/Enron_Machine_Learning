#!/usr/bin/python3

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import joblib

enron_data = joblib.load(open("../final_project/final_project_dataset.pkl", "rb"))
print(len(enron_data.keys()))
# print(len(enron_data['METTS MARK']))
c=0
for k,v in enron_data.items():
    if enron_data[k]['poi']==True:
        c+=1
print(c)
# print((enron_data.keys()))
# print(sorted(enron_data.keys()))
# print(enron_data['PRENTICE JAMES'])
# print(enron_data['LAY KENNETH L'])

c1=0;c2=0
for k,v in enron_data.items():
    if enron_data[k]['salary']!='NaN':
        c1+=1
    if enron_data[k]['email_address']!='NaN':
        c2+=1
print(c1,c2)
c3=0
for k,v in enron_data.items():
    if enron_data[k]['total_payments']=='NaN':
        c3+=1
print(c3)
# print(enron_data)
c4=0
c5=0
for k,v in enron_data.items():
    if enron_data[k]['poi']==True:
        c5+=1
        if enron_data[k]['total_payments']=='NaN':
            c4+1
print(c4,c5)



