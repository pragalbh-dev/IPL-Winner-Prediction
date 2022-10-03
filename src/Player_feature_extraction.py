#!/usr/bin/env python
# coding: utf-8

# # Player level feature extraction from raw database

# In[ ]:


import numpy as np
import pandas as pd
import os
dir=os.getcwd()
new_dir=dir[:-4]+'\data'
for y in range(10):
    df = pd.read_csv(new_dir+"\\"+str(y+2010)+"_scoreboard.csv")
    df=df.drop(columns=['Unnamed: 0'])
    df.loc[:,'Runs_Scored':'Nb']=df.loc[:,'Runs_Scored':'Nb'].astype(float)
    df=df.fillna(0)
    df2=pd.DataFrame(columns=list(df.columns)) 
    df2.Name=df.Name.unique()
    df2['50']=np.nan
    df2['100']=np.nan
    df2['0']=np.nan
    df2['4_haul']=np.nan
    df2['5_haul']=np.nan
    df2['BSR']=np.nan
    df2['BWA']=np.nan
    df2['hard_hit']=np.nan
#   df2['Avg']=np.nan     not out data not available
    df2['bat_consistency']=np.nan
    df2['pca_bat']=np.nan
    df2['Bowl_index']=np.nan
    df2['bowl_consistency']=np.nan
    df2['bowl_pca']=np.nan
    df2.loc[:,'Runs_Scored':'Nb']=df2.loc[:,'Runs_Scored':'Nb'].astype(float)
    df2=df2.fillna(0)
    i=0
    for player in df2.Name:
        df2.loc[i,'Runs_Scored':'Nb']=df[df.Name==player].loc[:,'Runs_Scored':'Nb'].sum(axis=0)
        df2.loc[i,'50']=len(df[df['Name']==player][df['Runs_Scored']>=50])
        df2.loc[i,'100']=len(df[df['Name']==player][df['Runs_Scored']>=100])
        df2.loc[i,'0']=len(df[df['Name']==player][df['Runs_Scored']==0])
        df2.loc[i,'4_haul']=len(df[df['Name']==player][df['Wickets']==4])
        df2.loc[i,'5_haul']=len(df[df['Name']==player][df['Wickets']>=5])
        i+=1
    df2['SR']=df2['Runs_Scored']/(df2['Balls_Faced']+1)*100
    df2['Econ']= df2['Runs_Given']/(df2['Overs_Bowled']+1)
    df2=df2.fillna(0)
    df2['hard_hit']=(df2['4s_hit']+df2['6s_hit'])/(df2['Balls_Faced']+1)
    
    df2['bat_consistency']=0.508*df2['SR']+0.25*df2['hard_hit']*100+0.082*df2['50']+0.05*df2['100']-0.04*df2['0']*5
    df2['pca_bat']=0.458*df2['Runs_Scored']+0.325*df2['SR']+0.406*df2['4s_hit']+0.415*df2['6s_hit']+0.432*df2['50']
    df2['BSR']=df2['Overs_Bowled']*6/(df2['Wickets']+1)
    df2['BWA']=df2['Runs_Given']/(df2['Wickets']+1)
    df2['Bowl_index']=df2['Econ']+df2['BSR']+df2['BWA']
    df2['bowl_consistency']=0.2*df2['Overs_Bowled']+0.3*(50/(df2['BWA']+1))+0.18*(50/(df2['SR']+1))+0.05*(df2['4_haul']+2*df2['5_haul'])*5+0.1*(df2['Maidens'])*6+0.05*20/(df2['Econ']+1)
    df2['bowl_pca']=-0.428*df2['Wickets']+0.591*df2['BWA']+0.383*df2['Econ']+0.566*df2['BSR']-0.2*df2['Dots']
    df2=df2.fillna(0)
#   df.replace([np.inf, -np.inf],500)
    df2.loc[:,'hard_hit':]=(df2.loc[:,'hard_hit':]-df2.loc[:,'hard_hit':].mean())/df2.loc[:,'hard_hit':].std() 
    df2.to_csv(new_dir+"\\"+str(y+2010)+"_player.csv",index=False)
                                         


# In[ ]:




