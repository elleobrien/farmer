import pandas as pd 
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import train_test_split
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import lightgbm as lgb
import neptune

from neptunecontrib.monitoring.lightgbm import neptune_monitor

import numpy as np
import pandas as pd

import pickle
 
import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


# In[2]:
neptune.init(api_token=os.getenv('NEPTUNE_API_TOKEN'),
            project_qualified_name=os.getenv('NEPTUNE_PROJECT_NAME'))

df = pd.read_csv('BostonData.csv',header=0)
#df.to_csv(r'C:\Users\hhurc\BostonData\BostonData.csv')


# ## EDA

# In[3]:


df.head(5)


# In[4]:


df_correl = df.corr()


# In[5]:


####import seaborn as sns
####import matplotlib.pyplot as plt
#plt.figure(figsize=(12,10))
#sns.heatmap(df_correl,annot=True)


# In[6]:


#df.info()


# In[7]:


#sns.pairplot(df)


# In[8]:


#standardize predictors
from sklearn.preprocessing import StandardScaler


# In[9]:


# standardize everything except CHAS and MEDV
features_stdz = list(set(df.columns) - {"CHAS","MEDV"})


# In[10]:


std_trans = StandardScaler()
df_trans = pd.DataFrame(std_trans.fit_transform(df[features_stdz]),columns=features_stdz)


# In[11]:


df0 = df_trans.merge(df[["CHAS","MEDV"]],right_index=True,left_index=True)


# In[12]:


df0


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


X = df0.iloc[:,0:13]
y = df0["MEDV"]


# In[15]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5)


neptune.init(api_token=os.getenv('NEPTUNE_API_TOKEN'),
            project_qualified_name=os.getenv('NEPTUNE_PROJECT_NAME'))

neptune.create_experiment('BostonData-NEPTUNE')




# In[16]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
model = reg.fit(X_train,y_train)
pickle.dump(model,open('model.pkl','wb'))

y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score
import math
mse = mean_squared_error(y_test, y_pred, squared=False)
rmse = math.sqrt(mse)
r2 = r2_score(y_test, y_pred)

neptune.log_metric('mse',mse)
neptune.log_metric('rmse', rmse)
neptune.log_metric('r2', r2)
# In[17]:


print("y-intercept",model.coef_[0])


# In[18]:

## UNCOMMENT FOR MLFLOW REPORTING
#mlflow.set_experiment(experiment_name="experiment1")
#mlflow.set_tracking_uri("http://localhost:5000")
#with mlflow.start_run():
#    mlflow.log_param("alpha1",model.coef_[0])
#    mlflow.log_param("beta1",model.coef_[1])


# In[ ]:

if os.getenv('CI') == "true":
    neptune.append_tag('ci-pipeline', os.getenv('NEPTUNE_EXPERIMENT_TAG_ID'))





with open("metrics.json", 'w') as outfile:
        json.dump({ "MSE": mse, "RMSE":rmse,"R2":r2 }, outfile)
        
pickle.dump(model,open('model.pkl','wb'))
# In[18]:

## UNCOMMENT FOR MLFLOW REPORTING
#mlflow.set_experiment(experiment_name="experiment1")
#mlflow.set_tracking_uri("http://localhost:5000")
#with mlflow.start_run():
#    mlflow.log_param("alpha1",model.coef_[0])
#    mlflow.log_param("beta1",model.coef_[1])
if os.getenv('CI') == "true":
    neptune.append_tag('ci-pipeline', os.getenv('NEPTUNE_EXPERIMENT_TAG_ID'))
