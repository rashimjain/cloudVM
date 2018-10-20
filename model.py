
# coding: utf-8

# In[3]:


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


# In[4]:


df = pd.read_csv('kiva_loans_20181020.csv')
df.head()


# In[5]:


df.shape


# In[6]:


df.status.value_counts()


# In[7]:


df.dtypes


# In[8]:


df.isnull().sum()


# In[9]:


df1 = df[['status','funded_amount', 'loan_amount', 'activity', 'sector',  'country',
         'currency','gender','term_in_months']]


# In[10]:


df1.head(2)


# In[11]:


df2 = df1.dropna()
df2 = df2.drop(['term_in_months', 'currency'], axis=1)
df2.head()


# In[12]:


df2.shape


# In[13]:


# Use Pandas get_dummies to convert categorical data

df2 = pd.get_dummies(df2)
df2.head()


# In[14]:


df2.shape


# In[15]:


X = df2.drop(['status', 'loan_amount', 'funded_amount'], axis=1)
y = df2['status']


# In[16]:


ss = StandardScaler()
lr = LogisticRegression()
lr_pipe = Pipeline([('sscale', ss), ('logreg', lr)])


# In[17]:


lr_pipe.fit(X, y)


# In[18]:


lr_pipe.score(X,y)


# # Divide the dataset into separate training (80% of the data) and test (20% of the data) datasets.

# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


# # Chain the StandardScaler and Logistic Regression objects in a pipeline.

# In[20]:


lr_pipe.fit(X_train, y_train)


# In[21]:


lr_pipe.score(X_test, y_test)  # prediction accuracy score


# In[22]:


lr_pipe.score(X_train, y_train)


# In[23]:


y_pred = lr_pipe.predict(X_test)


# In[24]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


# In[25]:


print(f1_score(y_test, y_pred, average="macro"))
print(precision_score(y_test, y_pred, average="macro"))
print(recall_score(y_test, y_pred, average="macro")) 


# # Alternative way of executing the Lograthmic Model. Lograthmic models don't require scaling.

# In[26]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[27]:


print(f"Training Data Score: {logmodel.score(X_train, y_train)}")
print(f"Testing Data Score: {logmodel.score(X_test, y_test)}")


# In[28]:


predictions = logmodel.predict(X_test)


# # 1 - Logistic Model Score

# In[29]:


from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))


# In[30]:


df4 = pd.DataFrame({"Prediction": predictions, "Actual": y_test}).reset_index(drop=True)


# In[31]:


df4.head(10)


# # Statistical Testing of the model for significance of independedent variables 

# In[32]:


import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm # conda install statsmodels - if there is an error
from scipy import stats



X2 = sm.add_constant(X_train)
est = sm.OLS(y_train, X2)
est2 = est.fit()
print(est2.summary())


# In[33]:


X2 = sm.add_constant(X_test)
est = sm.OLS(y_test, X2)
est2 = est.fit()
print(est2.summary())


# In[34]:


df_activity = df[['status', 'activity']]
df_activity = df_activity.dropna()
df_activity.head()


# In[35]:


df_activity.shape


# In[36]:


df_activity = pd.get_dummies(df_activity)
df_activity.head()


# In[37]:


X = df_activity.drop(['status'], axis=1)
y = df_activity['status']


# In[38]:


lm = LogisticRegression()
lm.fit(X,y)
params = np.append(lm.intercept_,lm.coef_)
predictions = lm.predict(X)

params = np.round(params,4)

myDF3 = pd.DataFrame()
index = [0]
params = np.delete(params, index)

myDF3["Activity_Feature_Name"],myDF3["Activity_Coefficients"] = [X.columns,params]
print(myDF3)


# In[88]:


keys = [i.replace('activity_', '') for i in df_activity.columns[1:]]
activity_features = dict(zip(keys, myDF3.Activity_Coefficients.values))

activity_coef = [activity_features[i] for i in df.activity.values]
df['activity_coef'] = activity_coef
df

