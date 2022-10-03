#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


import os
dir = os.getcwd()
new_dir = dir[:-4] + '\data'


# In[ ]:


new_dir


# In[ ]:


df = pd.read_csv(new_dir + '\\'+'final.csv')


# 

# In[ ]:


df = df.drop(['Unnamed: 0', 'Unnamed: 0.1','Time','Ump1','Ump2','Ump3','Ref','Venue'],axis=1)


# In[ ]:


df['Team1_F'] = df['Team1_F'] + 3
df['Team1_G'] += 3
df['Team1_H'] += 3
df['Team1_I'] += 3
df['Team1_J'] += 3
df['Team1_K'] += 3
df['Team2_F'] += 3
df['Team2_G'] += 3
df['Team2_H'] += 3
df['Team2_I'] += 3 
df['Team2_J'] += 3
df['Team2_K'] += 3


# In[ ]:


df


# In[ ]:


df1 = pd.get_dummies(df,columns = ['Team1','Team2'])


# In[ ]:


df1 = df1.drop(['Squad1','Squad2'],axis = 1)
df1


# In[ ]:


X = df1.drop(['Winner'],axis=1)


# In[ ]:


X_train = X[X.Year != 2019]
y_train = df1[df1.Year!=2019]['Winner']

X_test = X[X.Year == 2019]
y_test = df1[df1.Year == 2019]['Winner']


# In[ ]:


import matplotlib.pyplot as plt
f = plt.figure(figsize=(19, 15))
plt.matshow(X_train.corr(), fignum=f.number)
plt.xticks(range(X_train.shape[1]), X_train.columns, fontsize=14, rotation=45)
plt.yticks(range(X_train.shape[1]), X_train.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16);


# In[ ]:


X_train = X_train.drop(['Team1_G','Team2_G'],axis=1)
X_test = X_test.drop(['Team1_G','Team2_G'],axis=1)


# In[ ]:


X_train.shape


# In[ ]:


y_train.value_counts()


# In[ ]:


df1.shape


# In[ ]:


from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.fit_transform(X_test)


# In[ ]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(X_train,y_train).predict(X_test)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix  
print(classification_report(y_test,y_pred))


# In[ ]:


y_pred


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[ ]:


from sklearn.metrics import f1_score
f1_score(y_test,y_pred)


# In[ ]:


from imblearn.over_sampling import SMOTE
over = SMOTE(sampling_strategy=1)
X_train,y_train = over.fit_resample(X_train,y_train)


# In[ ]:


from sklearn.model_selection import GridSearchCV 
from sklearn.svm import SVC
param_grid2 = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  
  
grid2 = GridSearchCV(SVC(), param_grid2, refit = True, verbose = 3) 

# fitting the model for grid search 
grid2.fit(X_train, y_train)


# In[ ]:


y_pred2 = grid2.predict(X_test)


# In[ ]:


print(classification_report(y_test,y_pred2))


# In[ ]:


f1_score(y_test,y_pred2)


# In[ ]:


df2 = df[df['Year']==2019]


# In[ ]:


df2['Winner_pred2'] = y_pred2


# In[ ]:


df2


# In[ ]:


arr = np.array(df2['Team1'].unique())
df3 = pd.DataFrame(data = arr,columns = ['Teams'])


# In[ ]:


df3


# In[ ]:


df3['Wins_SVM'] = 0


# In[ ]:


df3


# In[ ]:


for ind in df2.index :
  if df2['Winner_pred2'][ind]==1 :
    for ind2 in df3.index :
      if df3['Teams'][ind2] == df2['Team1'][ind] :
        df3['Wins_SVM'][ind2] +=1
  else :
    for ind2 in df3.index :
      if df3['Teams'][ind2] == df2['Team2'][ind] :
        df3['Wins_SVM'][ind2] +=1


# In[ ]:


df3.drop(['Wins'],axis=1)


# In[ ]:




