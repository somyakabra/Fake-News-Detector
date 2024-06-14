#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
import string


# In[2]:


data_fake = pd.read_csv('Fake.csv')
data_true = pd.read_csv('True.csv')


# In[3]:


data_fake.head()


# In[4]:


data_true.head()


# In[5]:


data_fake["class"] =0
data_true['class'] =1


# In[6]:


data_fake.shape, data_true.shape


# In[7]:


data_fake_manual_testing = data_fake.tail(10)
for i in range(23480,23470,-1):
    data_fake.drop([i], axis = 0, inplace = True)
    
    
data_true_manual_testing = data_true.tail(10)
for i in range(21416,21406,-1):
    data_true.drop([i], axis = 0, inplace = True)    


# In[8]:


data_fake.shape, data_true.shape


# In[9]:


data_fake_manual_testing.head(10)


# In[10]:


data_true_manual_testing.head(10)


# In[11]:


data_merge = pd.concat([data_fake, data_true], axis = 0)
data_merge.head(10)


# In[12]:


data_merge.columns


# In[13]:


data = data_merge.drop(['title','subject','date'], axis = 1)


# In[14]:


data.isnull().sum()


# In[15]:


data = data.sample(frac = 1)


# In[16]:


data.head()


# In[17]:


data.reset_index(inplace = True)
data.drop(['index'], axis = 1, inplace = True)


# In[18]:


data.columns


# In[19]:


data.head()


# In[20]:


def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


# In[ ]:


data['text'] = data['text'].apply(wordopt)


# In[22]:


x = data['text']
y = data['class']


# In[23]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.25)


# In[24]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)


# # Logistic Regression

# In[25]:


from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(xv_train, y_train)


# In[26]:


pred_lr = LR.predict(xv_test)


# In[27]:


LR.score(xv_test, y_test)


# In[28]:


print(classification_report(y_test, pred_lr))


# # Decision Tree Classifier
# 

# In[30]:


from sklearn.tree import DecisionTreeClassifier

DT  = DecisionTreeClassifier()
DT.fit(xv_train, y_train)


# In[31]:


pred_dt = DT.predict(xv_test)


# In[32]:


DT.score(xv_test, y_test)


# In[33]:


print(classification_report(y_test, pred_lr))


# # GradientBoostingClassifier

# In[39]:


from sklearn.ensemble import GradientBoostingClassifier

GB = GradientBoostingClassifier(random_state = 0)
GB.fit(xv_train, y_train)


# In[ ]:


pred_rf = RF.predict(xv_test)


# In[ ]:


RF.score(xv_test, y_test)


# In[ ]:


print(classification_report(y_test, pred_rf))


# In[ ]:


news = str(input())
manual_testing(news)


# In[ ]:


def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"
    
def manual_testion(news):
    testing_news = {"text":[news]}
    new_def_test = pd.dataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorizatioin.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GBC = GBC.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)
    
    return print("\n\nLR Prediction: {} \nDT prediction: {} \nGB Prediction: {} \nRFC Prediction: {}".format(output_lable(pred_LR[0]),  
                                                                                                             output_lable(pred_GBC[0]),
                                                                                                             output_lable(pred_RFC[0])))


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier

GB = GradientBoostingClassifier(random_state = 0)
GB.fit(xv_train, y_train)


# In[ ]:




