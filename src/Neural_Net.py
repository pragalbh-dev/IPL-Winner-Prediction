#!/usr/bin/env python
# coding: utf-8

# # Neural Network model

# In[ ]:


import numpy as np
import os
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
sc=StandardScaler()
np.random.seed(31415)
dir=os.getcwd()


# In[ ]:


new_dir=dir[:-4]+'\data'


# Addition of home column and seperating test train data

# In[ ]:


dff=pd.read_csv(new_dir+'\db2.csv')
dfft=pd.read_csv(new_dir+'\db2_test.csv')
dff['Home']=np.nan
dfft['Home']=np.nan
def common_member(a, b): 
    a_set = set(a) 
    b_set = set(b) 
    if (a_set & b_set): 
        return True 
    else: 
        return False
home={'CSK':['Chennai'],'KKR':['Kolkata'],'DC':['Delhi'],'DCG':['Hyderabad'],'GL':['Ahmedabad','Rajkot'],'KTK':['Kochi'],'MI':['Mumbai'],'KXIP':['Mohali'],'RCB':['Bengaluru'],'RR':['Jaipur'],'PWI':['Pune'],'SRH':['Hyderabad'],'RPS':['Pune']}
i=0
for ven in dff['Venue']:
    v=ven.split()
    if common_member(home[dff.loc[i,'Team1']],v):
        dff.loc[i,'Home']=1
    elif common_member(home[dff.loc[i,'Team2']],v):
        dff.loc[i,'Home']=0
    else :
        dff.loc[i,'Home']=2
    i=i+1
i=0
for ven in dfft['Venue']:
    v=ven.split()
    if common_member(home[dfft.loc[i,'Team1']],v):
        dfft.loc[i,'Home']=1
    elif common_member(home[dfft.loc[i,'Team2']],v):
        dfft.loc[i,'Home']=0
    else :
        dfft.loc[i,'Home']=2
    i=i+1
dff1=dff.drop(columns=['Unnamed: 0','Year','Team1','Team2','Venue','Time','Ump1','Ump2','Ump3','Ref','Squad1','Squad2'])
dff1t=dfft.drop(columns=['Unnamed: 0','Year','Team1','Team2','Venue','Time','Ump1','Ump2','Ump3','Ref','Squad1','Squad2'])
Y_train=dff1.Winner.to_numpy()
dff1=dff1.drop(columns=['Winner'])
Y_test=dff1t.Winner.to_numpy()
dff1t=dff1t.drop(columns=['Winner'])
X_train=dff1.values
X_test=dff1t.values
X_train2=X_train+3
X_test2=X_test+3
X_train = sc.fit_transform(X_train2)
X_test=sc.transform(X_test2)


# First Run with rando structure and parameters

# In[ ]:



# X_val2=sc.transform(X_val2)
# Y_train=y_smt
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
clf = MLPClassifier(learning_rate_init=0.001,random_state=21,solver='adam',activation='relu',warm_start=True,
                    learning_rate='adaptive',max_iter=500,hidden_layer_sizes=(10,10,10),alpha=0.1,tol=0.001)
clf.fit(X_train,Y_train)
y_pred = clf.predict(X_train)
y_testpred=clf.predict(X_test)
# scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']
# scores = cross_validate(clf, X_train, y_train, scoring=scoring, cv=2)
# scores2 = cross_validate(clf, X_test, y_test, scoring=scoring, cv=2)
print(accuracy_score(Y_train,y_pred),accuracy_score(Y_test,y_testpred))
print(f1_score(Y_train,y_pred),f1_score(Y_test,y_testpred))


# First grid search to find optimum Structure or number of layers and get a rough iea of number of units in each layer

# In[ ]:


params={'alpha': 10.0 **(-np.arange(1, 10)),'learning_rate_init':[0.001,0.005,0.008,0.0005],'activation':['logistic', 'tanh', 'relu'],'random_state':[0,1,2,3,4,5,6,7,8,9],'hidden_layer_sizes':[(10,10,5),(10,10,10,5),(10,10,10,10,5),(10,10,10,10,10,5),(50,20,10),(30,30,30),(100,100,100),(50,50,50,50)]}


# In[ ]:


gs=GridSearchCV(clf,params)
gs.fit(X_train, Y_train)
dt=pd.DataFrame(gs.cv_results_)
dt.to_csv(new_dir+"\score.csv",index=False)


# Second grid Search after shorlisting the number of layers to 3 and number of units around 10

# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
params2={'learning_rate':['invscaling','adaptive'],'alpha': [5,4,3,2,1,0.1,0.001,0.01,0.005,0.0001],
         'learning_rate_init':[0.1,0.05,0.001,0.005,0.008,0.0005,0.01],'random_state':[0,1,2,3,4,5,6,7,8,9],
         'hidden_layer_sizes':[(4,4,4),(6,6,6),(10,10,10),(8,8,8),(15,15,15),(10,6,4),(5,4,4),(10,4,6),(12,6,4)],
         'activation':['relu','tanh','logistic'],'solver':['adam','sgd']}
scoring2 = {'f1': 'f1_weighted', 'Accuracy': make_scorer(accuracy_score)}


# In[ ]:


gs2=GridSearchCV(clf,params2,scoring=scoring2,refit='f1',return_train_score=True,n_jobs=-1)
gs2.fit(X_train, Y_train)
dt2=pd.DataFrame(gs2.cv_results_)
dt2.to_csv(new_dir+"\score3.csv",index=False)


# Third and final grid search to get the hyperparameters in a narrowed space and after selection the structure as (10,6,4)

# In[ ]:


params3={'learning_rate':['adaptive'],'alpha': [1.2,1,0.5,0.2,0.3,0.1,0.001,0.002,0.01,0.005,0.0001],
         'learning_rate_init':[0.001,0.005,0.008,0.0005,0.01],'random_state':[0,13,41,52,89,148,185,210],
         'hidden_layer_sizes':[(10,6,4)],'tol':[0.005,0.008,0.003,0.0025,0.001,0.0001,0.0003,0.0004],
         'activation':['relu']}
gs3=GridSearchCV(clf,params3,scoring=scoring2,refit='f1',return_train_score=True,n_jobs=-1)
gs3.fit(X_train, Y_train)
dt2=pd.DataFrame(gs3.cv_results_)
dt2.to_csv(new_dir+"\score6.csv",index=False)


# From results of the previous run we reached at the following hyperparameters 
# The best case was obviosly overfitting due to less data and a complex model for the problem but we saved the gid search results in an excel file and then mapped the test and train accuracy values for all the grid points to a value that is high if the difference in train and test accuracies are low and high if the test accuracy is high. doing that we reached our final set of hyperparameters

# In[ ]:


X_train0=X_train
Y_train0=Y_train
clf2 = MLPClassifier(learning_rate_init=0.0015,random_state=185,solver='adam',activation='relu',warm_start=True,
                    learning_rate='adaptive',max_iter=3000,hidden_layer_sizes=(10,6,4),alpha=0.5,tol=0.0025)
clf2.fit(X_train0,Y_train0)
y_testpred2=clf2.predict(X_test)
print(classification_report(Y_test,y_testpred2))


# In[ ]:




