#!/usr/bin/env python
# coding: utf-8

# # Converting categorical columns to binary

# In[ ]:

import os
import pandas as pd
dir=os.getcwd()
new_dir=dir[:-4]+'\data'
df9=pd.read_csv(new_dir+"\DB_1.csv")
df9['Winner']=(df9['Winner']==df9['Team1'])*1
df9['Toss_won']=(df9['Toss_won']==df9['Team1'])*1
df9['Decision']=(df9['Decision']=='bowl')*1
df9.to_csv(new_dir+"\DB.csv",index=False)

