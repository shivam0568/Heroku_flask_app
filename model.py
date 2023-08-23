#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('C:/Users/hp/PycharmProjects/flaskProject/yield_df.csv')


# In[ ]:


df.head()


# In[4]:


df.drop('Unnamed: 0',axis=1,inplace=True)


# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


df.info()


# In[8]:


df.isnull().sum()


# In[ ]:


df.duplicated().sum()


# In[ ]:


df.drop_duplicates(inplace=True)


# In[ ]:


df.duplicated().sum()


# # Transforming average_rain_fall_mm_per_year
# In summary, this code identifies the indices of rows in the DataFrame df where the values in the column 'average_rain_fall_mm_per_year' are not numeric strings. These rows can be considered for removal or further processing, depending on the specific use case.

# In[ ]:


def isStr(obj):
    try:
        float(obj)
        return False
    except:
        return True
to_drop = df[df['average_rain_fall_mm_per_year'].apply(isStr)].index


# In[ ]:


df = df.drop(to_drop)


# In[ ]:


df


# In[ ]:


df['average_rain_fall_mm_per_year'] = df['average_rain_fall_mm_per_year'].astype(np.float64)


# # Graph Frequency vs Area

# In[ ]:


len(df['Area'].unique())


# In[ ]:


plt.figure(figsize=(15,20))
sns.countplot(y=df['Area'])
plt.show()


# In[ ]:


(df['Area'].value_counts() < 500).sum()


# # yield_per_country

# In[ ]:


country = df['Area'].unique()
yield_per_country = []
for state in country:
    yield_per_country.append(df[df['Area']==state]['hg/ha_yield'].sum())


# In[ ]:


df['hg/ha_yield'].sum()


# In[ ]:


yield_per_country


# # Yield Per Country Graph

# In[ ]:


plt.figure(figsize=(15, 20))
sns.barplot(y=country, x=yield_per_country)


# # Graph Frequency vs Item

# In[ ]:


sns.countplot(y=df['Item'])


# # Yield Vs Item

# In[ ]:


crops = df['Item'].unique()
yield_per_crop = []
for crop in crops:
    yield_per_crop.append(df[df['Item']==crop]['hg/ha_yield'].sum())


# In[ ]:


sns.barplot(y=crops,x=yield_per_crop)


# # Train Test split Rearranging Columns

# In[ ]:


col = ['Year', 'average_rain_fall_mm_per_year','pesticides_tonnes', 'avg_temp', 'Area', 'Item', 'hg/ha_yield']
df = df[col]
X = df.iloc[:, :-1]
y = df.iloc[:, -1]


# In[ ]:


df.head(3)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0, shuffle=True)


# # Converting Categorical to Numerical and Scaling the values

# In[ ]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
ohe = OneHotEncoder(drop='first')
scale = StandardScaler()

preprocesser = ColumnTransformer(
        transformers = [
            ('StandardScale', scale, [0, 1, 2, 3]),
            ('OHE', ohe, [4, 5]),
        ],
        remainder='passthrough'
)


# In[ ]:


X_train_dummy = preprocesser.fit_transform(X_train)
X_test_dummy = preprocesser.transform(X_test)


# In[ ]:


preprocesser.get_feature_names_out(col[:-1])


# # Let's train our model

# In[ ]:


#linear regression
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error,r2_score


models = {
    'lr':LinearRegression(),
    'lss':Lasso(),
    'Rid':Ridge(),
    'Dtr':DecisionTreeRegressor()
}
for name, md in models.items():
    md.fit(X_train_dummy,y_train)
    y_pred = md.predict(X_test_dummy)
    
    print(f"{name} : mae : {mean_absolute_error(y_test,y_pred)} score : {r2_score(y_test,y_pred)}")


# # Select model

# In[ ]:


dtr = DecisionTreeRegressor()
dtr.fit(X_train_dummy,y_train)
dtr.predict(X_test_dummy)


# # Predictive System

# In[ ]:


def prediction(Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item):
    # Create an array of the input features
    features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)

    # Transform the features using the preprocessor
    transformed_features = preprocesser.transform(features)

    # Make the prediction
    predicted_yield = dtr.predict(transformed_features).reshape(1, -1)

    return predicted_yield[0]

Year = 1990
average_rain_fall_mm_per_year =1485.0
pesticides_tonnes = 121.00
avg_temp = 16.37                   
Area = 'Albania'
Item = 'Maize'
result = prediction(Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item)


# In[ ]:


result


# In[ ]:


1990	1485.0	121.00	16.37	Albania	Maize	36613
2013	657.0	2550.07	19.76	Zimbabwe	Sorghum	3066


# # Pickle Files

# In[ ]:


import pickle
pickle.dump(dtr,open('dtr.pkl','wb'))
pickle.dump(preprocesser,open('preprocessor.pkl','wb'))


# In[3]:


import sklearn
print(sklearn.__version__)


# In[ ]:




