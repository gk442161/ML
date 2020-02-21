#!/usr/bin/env python
# coding: utf-8

# In[128]:





# In[129]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split




# In[130]:


data=pd.read_csv('winequality-red.csv',sep=';')


# In[131]:


data.shape


# In[132]:


x_train,x_test,y_train,y_test=train_test_split(data,data['quality'],test_size=0.3,train_size=0.7,random_state=1)


# In[133]:


x_train.shape,x_test.shape


# In[134]:


dtc=DecisionTreeClassifier()
dtc=dtc.fit(x_train,y_train)
y_dtc_pred=dtc.predict(x_test)


# In[135]:


print("Decision_Tree Accuracy:",accuracy_score(y_test, y_dtc_pred))


# In[136]:


gnb = GaussianNB()
y_gnb_pred = gnb.fit(x_train, y_train).predict(x_test)


# In[137]:


print("gaussian_naive_bayes Accuracy:",accuracy_score(y_test, y_gnb_pred))


# In[138]:


rfc = RandomForestClassifier(max_depth=2, random_state=0,n_estimators=100)
rfc.fit(x_train,y_train)


# In[139]:


print(rfc.feature_importances_)


# In[140]:


y_rfc_pred=rfc.predict(x_test)


# In[141]:


print("Random_Forest_Classifier Accuracy:",accuracy_score(y_test, y_rfc_pred))


# In[142]:


lrc = LogisticRegression(random_state=0).fit(x_train, y_train)
y_lrc_pred=lrc.predict(x_test)


# In[143]:


print("Logistic_Regression Accuracy:",accuracy_score(y_test,y_lrc_pred))


# In[148]:


test_methods=np.array(['Decision_Tree','gaussian_naive_bayes','Random_Forest_Classifier Accuracy','Logistic_Regression Accuracy'])


# In[149]:


test_accuracy=np.array([accuracy_score(y_test,y_dtc_pred),accuracy_score(y_test,y_gnb_pred),accuracy_score(y_test,y_rfc_pred),accuracy_score(y_test,y_lrc_pred)])


# In[165]:


rng = np.random.RandomState(0)
colors = rng.rand(4)
plt.style.use('seaborn-whitegrid')
plt.scatter(test_methods,test_accuracy,c=colors,s=1000,cmap='viridis',alpha=0.25)
plt.ylabel('ACCURACY')
plt.xlabel('TESTS')
plt.figure(figsize=(10, 8), dpi=80)
plt.show()


# In[ ]:




