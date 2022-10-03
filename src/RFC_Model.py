#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score
from sklearn import preprocessing
from sklearn.metrics import classification_report 


# In[4]:


dir = os.getcwd()

new_dir = dir[:-4] + '\\data\\'

train = pd.read_csv(new_dir + 'DB_2.csv')

fields = list(train.columns)

fields.remove('Squad1')
fields.remove('Squad2')


norm_cols = list(train.select_dtypes(include='float64').columns)


train = train[ fields ]

for column in train.columns:
    if column  in norm_cols:
        train[column] = (train[column] - train[column].min()) / (train[column].max() - train[column].min())


# In[5]:


cols = list( train.select_dtypes(include='object').columns )

cols.append('Form1')
cols.append('Form2')

fcol = list(train.columns)

fcol.remove('Winner')

train_df = train[fcol]


# In[6]:


#train.head()

train_df=pd.get_dummies(train_df, columns= cols)


# In[7]:


train_df.head()

x_train = train_df[ train_df.Year != 2019 ]
y_train = train[ train.Year != 2019 ][ 'Winner' ]

x_test = train_df[ train_df.Year == 2019 ]
y_test = train[ train.Year == 2019 ][ 'Winner' ]

#print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)


# In[8]:


count = 0

for val in y_test:
    if val == 0:
        count += 1

a = count / len(y_test) 

b = 1 - a

print('0: ',a,'1: ',b)


# In[9]:


rfc=RandomForestClassifier(random_state=42)


# In[10]:


param_grid = { 
    'n_estimators': [200, 500, 700],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}


# In[11]:


CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(x_train, y_train)


# In[12]:


CV_rfc.best_params_


# In[13]:


rfc1=RandomForestClassifier(random_state=42, max_features='log2', n_estimators= 200, max_depth=6, criterion='entropy')


# In[14]:


rfc1.fit(x_train, y_train)


# In[15]:


pred=rfc1.predict(x_test)

print(pred)


# In[16]:


print(classification_report(y_test,pred))


# In[17]:


print(f1_score(y_test,pred))
print(accuracy_score(y_test,pred))


# In[ ]:




