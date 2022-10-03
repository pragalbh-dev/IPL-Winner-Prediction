#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import metrics

import itertools

from matplotlib.ticker import NullFormatter

import matplotlib.ticker as ticker



from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)

import os
dir=os.getcwd()
new_dir=dir[:-4]+'\data'



churn_df = pd.read_csv(new_dir + "\DB_2.csv")
churn_df.head()

churn_df.Venue.unique()

dict={'M.Chinnaswamy Stadium, Bengaluru':'RCB',
      'Punjab Cricket Association IS Bindra Stadium, Mohali':'KXIP',
      'Arun Jaitley Stadium, Delhi':'DC',
      'Eden Gardens, Kolkata':'KKR',
      'Wankhede Stadium, Mumbai':'MI',
      'Sawai Mansingh Stadium, Jaipur':'RR',
      'Rajiv Gandhi International Stadium, Hyderabad':'SRH',
      'MA Chidambaram Stadium, Chennai':'CSK',
      'Dr DY Patil Sports Academy, Mumbai':'MI',
      'Brabourne Stadium, Mumbai':'MI',
      'Sardar Patel Stadium, Ahmedabad':'GL',
      'Nehru Stadium, Kochi':'KTK',
      'Maharashtra Cricket Association Stadium, Pune':'RPS',
      'Saurashtra Cricket Association Stadium, Rajkot':'GL',
      'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam':'DCG'
      
     }
churn_df['Venue']= churn_df['Venue'].map(dict)
churn_df.head()

for index in churn_df.index:
    if churn_df.loc[index,'Venue']==churn_df.loc[index,'Team1']:
        churn_df.loc[index,'Venue']=1
    elif churn_df.loc[index,'Venue']==churn_df.loc[index,'Team2']:
        churn_df.loc[index,'Venue']=2
    else:
        churn_df.loc[index,'Venue']=3
        
for index in churn_df.index:
    if churn_df.loc[index,'Team1']=='RCB':
        churn_df.loc[index,'Team1']=1
    elif churn_df.loc[index,'Team1']=='KXIP':
        churn_df.loc[index,'Team1']=2
    elif churn_df.loc[index,'Team1']=='DC':
        churn_df.loc[index,'Team1']=3
    elif churn_df.loc[index,'Team1']=='KKR':
        churn_df.loc[index,'Team1']=4
    elif churn_df.loc[index,'Team1']=='MI':
        churn_df.loc[index,'Team1']=5
    elif churn_df.loc[index,'Team1']=='RR':
        churn_df.loc[index,'Team1']=6
    elif churn_df.loc[index,'Team1']=='DCG':
        churn_df.loc[index,'Team1']=7
    elif churn_df.loc[index,'Team1']=='CSK':
        churn_df.loc[index,'Team1']=8
    elif churn_df.loc[index,'Team1']=='KTK':
        churn_df.loc[index,'Team1']=9
    elif churn_df.loc[index,'Team1']=='PWI':
        churn_df.loc[index,'Team1']=10
    elif churn_df.loc[index,'Team1']=='SRH':
        churn_df.loc[index,'Team1']=11
    elif churn_df.loc[index,'Team1']=='GL':
        churn_df.loc[index,'Team1']=12
    elif churn_df.loc[index,'Team1']=='RPS':
        churn_df.loc[index,'Team1']=13

for index in churn_df.index:
    if churn_df.loc[index,'Team2']=='RCB':
        churn_df.loc[index,'Team2']=1
    elif churn_df.loc[index,'Team2']=='KXIP':
        churn_df.loc[index,'Team2']=2
    elif churn_df.loc[index,'Team2']=='DC':
        churn_df.loc[index,'Team2']=3
    elif churn_df.loc[index,'Team2']=='KKR':
        churn_df.loc[index,'Team2']=4
    elif churn_df.loc[index,'Team2']=='MI':
        churn_df.loc[index,'Team2']=5
    elif churn_df.loc[index,'Team2']=='RR':
        churn_df.loc[index,'Team2']=6
    elif churn_df.loc[index,'Team2']=='DCG':
        churn_df.loc[index,'Team2']=7
    elif churn_df.loc[index,'Team2']=='CSK':
        churn_df.loc[index,'Team2']=8
    elif churn_df.loc[index,'Team2']=='KTK':
        churn_df.loc[index,'Team2']=9
    elif churn_df.loc[index,'Team2']=='PWI':
        churn_df.loc[index,'Team2']=10
    elif churn_df.loc[index,'Team2']=='SRH':
        churn_df.loc[index,'Team2']=11
    elif churn_df.loc[index,'Team2']=='GL':
        churn_df.loc[index,'Team2']=12
    elif churn_df.loc[index,'Team2']=='RPS':
        churn_df.loc[index,'Team2']=13
    
#churn_df


churn_df = churn_df[[ 'Year', 'Team1', 'Team2', 'Toss_won', 'Decision', 'Team_1_A', 'Team_1_B', 'Team_1_C',
       'Team_1_D', 'Team_1_E', 'Team1_F', 'Team1_G', 'Team1_H', 'Team1_I',
       'Team1_J', 'Team1_K', 'Team_2_A', 'Team_2_B', 'Team_2_C', 'Team_2_D',
       'Team_2_E', 'Team2_F', 'Team2_G', 'Team2_H', 'Team2_I', 'Team2_J',
       'Team2_K', 'Form1', 'Form2', 'Winner', 'Team_1_Balance',
       'Team_2_Balance']]
churn_df
X = churn_df[['Year', 'Team1', 'Team2', 'Toss_won', 'Decision', 'Team_1_A', 'Team_1_B', 'Team_1_C',
       'Team_1_D', 'Team_1_E', 'Team1_F', 'Team1_G', 'Team1_H', 'Team1_I',
       'Team1_J', 'Team1_K', 'Team_2_A', 'Team_2_B', 'Team_2_C', 'Team_2_D',
       'Team_2_E', 'Team2_F', 'Team2_G', 'Team2_H', 'Team2_I', 'Team2_J',
       'Team2_K', 'Form1', 'Form2', 'Team_1_Balance',
       'Team_2_Balance']]

y = churn_df['Winner']


#from sklearn import preprocessing
#X = preprocessing.StandardScaler().fit(X).transform(X)
#X[0:5]

from sklearn.model_selection import train_test_split
X_train=X[X.Year!=2019]
X_test=X[X.Year==2019]
y_train=churn_df[churn_df.Year!=2019]['Winner']
y_test=churn_df[churn_df.Year==2019]['Winner']

from imblearn.over_sampling import SMOTE 
sm = SMOTE(random_state = 2) 
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train_res,y_train_res)
LR

yhat = LR.predict(X_test)
yhat

yhat_prob = LR.predict_proba(X_test)
yhat_prob

#from sklearn.metrics import jaccard_similarity_score
#jaccard_similarity_score(y_test, yhat)


from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, yhat, labels=[1,0]))

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Winner=1','Winner=0'],normalize= False,  title='Confusion matrix')

print (classification_report(y_test, yhat))

from sklearn.metrics import log_loss
log_loss(y_test, yhat_prob)

fpr, tpr, thresholds = roc_curve(y_test, yhat)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.001])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show();


# In[16]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc

plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 


# In[17]:


from sklearn.neighbors import KNeighborsClassifier
k = 9
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh

yhat = neigh.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, yhat, labels=[1,0]))

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Winner=1','Winner=0'],normalize= False,  title='Confusion matrix')

print (classification_report(y_test, yhat))

from sklearn.metrics import log_loss
log_loss(y_test, yhat_prob)

fpr, tpr, thresholds = roc_curve(y_test, yhat)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.001])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show();


# In[ ]:





# In[ ]:





# In[ ]:




