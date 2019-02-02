
# coding: utf-8

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[3]:


cd ../input


# In[4]:


ls


# In[5]:


# Import Dependencies
import pandas as pd
import seaborn as sb
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np


# In[6]:


# Read the Train and Test File
df_train= pd.read_csv("train_V2.csv")
df_test= pd.read_csv("test_V2.csv")


# In[7]:


df_train.describe()


# In[8]:


import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize=(15, 15))
sb.heatmap(df_train.corr(), annot=True, fmt='.2f', ax=ax)


# In[9]:


# Data Preprocessing
LabEnc = LabelEncoder()
enc_train = df_train['matchType'].unique()
enc_test = df_test['matchType'].unique()
encode_train = LabEnc.fit(enc_train)
encode_test = LabEnc.fit(enc_test)
df_train['matchType'] = encode_train.transform(df_train['matchType'])
df_test['matchType'] = encode_test.transform(df_test['matchType'])


# In[10]:


# Deals with NaN values
df_train.iloc[:,-1] = df_train.iloc[:,-1].fillna(np.mean(df_train.iloc[:,-1]))


# In[11]:


from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(df_train.iloc[:,3:-1],df_train.iloc[:,-1], test_size=0.4)


# In[12]:


from sklearn.ensemble import RandomForestRegressor

RanFor = RandomForestRegressor(n_jobs=4, n_estimators=10)
RanFor.fit(Xtrain, Ytrain)

Y_RanFor = RanFor.predict(Xtest)
Test_MSE_RFR = mean_squared_error(Ytest,Y_RanFor)
print(('Random Forest training testset score: {s:.3f}').format(s=Test_MSE_RFR))


# In[13]:


X_submit=df_test
X_submit1=df_test.iloc[:,3:28]
df_submit = X_submit[['Id', 'matchId', 'groupId']]
Y_submit = RanFor.predict(X_submit1)
df_submit['prediction'] = Y_submit


# In[14]:


os.chdir("/kaggle/working/")


# In[ ]:


df_test2 = pd.read_csv('../input/sample_submission_V2.csv')
df_test2['winPlacePerc'] = df_submit['prediction'].copy()

df_test2.to_csv('submission_IV.csv', index=False) 
print('Random Forest submission file made\n')

