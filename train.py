import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

###

df = pd.read_csv('BostonData.csv',header=0)






# In[4]:


df_correl = df.corr()


# In[5]:


####import seaborn as sns
####import matplotlib.pyplot as plt
#plt.figure(figsize=(12,10))
#sns.heatmap(df_correl,annot=True)


sns.set_color_codes("dark")
ax = sns.heatmap(df_correl,annot=True)
plt.savefig("by_region.png",dpi=80)



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


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


# In[16]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
model = reg.fit(X_train,y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score
import math
mse = mean_squared_error(y_test, y_pred, squared=False)
rmse = math.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# In[17]:


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

