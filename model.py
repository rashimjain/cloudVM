
# coding: utf-8

# In[1]:


import re
import pandas as pd
import numpy as np
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


df = pd.read_csv('kiva_loans_20181016_20percent data.csv')
df.head(300)


# In[3]:


df.shape


# In[4]:


df.status.value_counts()


# In[5]:


df.dtypes


# In[6]:


df.isnull().sum()


# In[7]:


df1 = df[['status','funded_amount', 'loan_amount', 'activity', 'sector',  'country',
         'currency','gender','term_in_months']]


# In[8]:


df1.head(2)


# In[9]:


df2 = df1.dropna()
df2 = df2.drop(['term_in_months', 'currency'], axis=1)
df2.head()


# In[10]:


df2.shape


# In[11]:


# Use Pandas get_dummies to convert categorical data

df2 = pd.get_dummies(df2)
df2.head()


# In[12]:


df2.shape


# In[13]:


X = df2.drop(['status', 'loan_amount', 'funded_amount'], axis=1)
feature_names = X.columns
y = df2['status']


# # The k-nearest neighbors algorithm (KNN)

# In[14]:


from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# In[15]:


# Loop through different k values to see which has the highest accuracy
# Note: We only use odd numbers because we don't want any ties
train_scores = []
test_scores = []
a = 3000
b = 6000
for k in range(a, b, 500):
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn.fit(X_train, y_train)
    train_score = knn.score(X_train, y_train)
    test_score = knn.score(X_test, y_test)
    train_scores.append(train_score)
    test_scores.append(test_score)
    print(f'k: {k}, Train/Test Score: {train_score:.3f}/{test_score:.3f}')
    
    
plt.plot(range(a, b, 500), train_scores, marker='o')
plt.plot(range(a, b, 500), test_scores, marker="x")
plt.xlabel("k neighbors")
plt.ylabel("Testing accuracy Score")
plt.show()


# In[ ]:


# Note that k: XXXX seems to be the best choice for this dataset
knn = KNeighborsClassifier(n_neighbors=8000)
knn.fit(X_train, y_train)
print('k=8000 Test Acc: %.3f' % knn.score(X_test, y_test))


# In[17]:


predictions = knn.predict(X_test)


# In[18]:


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


# In[19]:


df4 = pd.DataFrame({"Prediction": predictions, "Actual": y_test}).reset_index(drop=True)


# In[20]:


df4.head(50)


# # Passing Real Time Feature Data for Testing on the Model.

# In[ ]:


inputs = {'country_India' : 1, 'gender_male' : 1, 'activity_Agriculture' : 1}

test = pd.Series(index=df2.columns)
for key in inputs.keys():
    test[key] = inputs[key]
    
test.fillna(0, inplace=True)


# In[ ]:


test1 = test.drop(['status','loan_amount', 'funded_amount'])

predictions = knn.predict_proba(test1.values.reshape(1, -1))
print (predictions)

