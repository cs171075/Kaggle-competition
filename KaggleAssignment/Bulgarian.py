#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import sklearn.preprocessing as skp
import sklearn.model_selection as skm
#import classification modules
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score, recall_score, roc_auc_score,roc_curve, auc, f1_score


# In[3]:


import os
os.chdir("C:\Users\HP\Desktop\jalil")


# In[4]:


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
print("Shape of train is: ", train.shape)
print("Shape of test is: ", test.shape)


# In[5]:


y = train['label'].copy()
X = train.drop(['label'],axis=1)
features = X.columns


# In[6]:


# bhai pehlay train se tarin karo aur test se label save karao


# In[7]:


trainX, testX, trainy, testy= skm.train_test_split(X,y, test_size=0.15, random_state=99) #explain random state
print("\n shape of train split: ")
print(trainX.shape, trainy.shape)
print("\n shape of train split: ")
print(testX.shape, testy.shape)


# In[11]:


from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(trainX,trainy)
predictions = gnb.predict(testX)
accgnb=accuracy_score(testy, predictions)*100
print("Accuracy of Gaussian Naive Bayes (%): \n",accgnb)  


# In[13]:


from sklearn.neural_network import MLPClassifier
nn=MLPClassifier()
nn.fit(trainX,trainy)
predictions = nn.predict(testX)
accnn=accuracy_score(testy, predictions)*100
print("Accuracy of Neural Networks (%): \n",accnn)


# In[21]:


svm=clf = SVC(gamma="auto",kernel='poly',degree=3)
svm.fit(trainX,trainy)
predictions = svm.predict(testX)
accsvm=accuracy_score(testy, predictions)
print("Accuracy of Support Vector Machine (%): \n",accsvm)  


# In[22]:


predictions_df=pd.DataFrame(predictions)


# In[23]:


predictions_df.columns=['Label']


# In[24]:


# predictions
predictions_df


# In[25]:


predictions_df.to_csv('prediction.csv')


# In[ ]:




