# %% [markdown]
# *Import necessary libraries*

# %%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor 
from sklearn import metrics 

import warnings 
warnings.filterwarnings('ignore')

# %% [markdown]
# *Loading the train and test dataset*

# %%
Train_data = pd.read_csv("bigmart_train.csv")
Test_data = pd.read_csv("bigmart_test.csv")

# %%
Train_data.head()

# %%
Test_data.head()

# %% [markdown]
# *Checking the shape of the datasets*

# %%
print('Train data:', Train_data.shape)
print('Test data:', Test_data.shape)

# %% [markdown]
# *Description of the data*

# %%
Train_data.describe().T

# %% [markdown]
# *Checking null values in train and test data*

# %%
Train_data.isnull().sum()

# %%
Test_data.isnull().sum()

# %% [markdown]
# *Concatenate the train and test data*

# %%
# create a new column if it is train data the column tag is "train", if it is test data the tag is "test"
Train_data['source'] = 'train'
Test_data['source'] = 'test'
df = pd.concat([Train_data, Test_data], ignore_index=True)

# %%
df.head()

# %%
df.tail()

# %%
# Check null values again
df.isnull().sum()

# %%
df.shape

# %% [markdown]
# ## **Visualization**

# %%
for i in Train_data.describe().columns:
    sns.distplot(Train_data[i].dropna(), color='blue')
    plt.show()

# %% [markdown]
# - most of the item weight between 5 and 20.
# - most of the item visibility is very low.
# - in 1990 there is no establishment.
# - most of the item outlet sales are between 0 and 4000.

# %%
for i in Train_data.describe().columns:
    sns.boxplot(Train_data[i].dropna())
    plt.show()

# %%
# visualize the item type and distribution of each
plt.figure(figsize=(15,10))
sns.countplot(Train_data['Item_Type'])
plt.xticks(rotation=90)
plt.show()

# %% [markdown]
# - most of the people comes and buy Fruits and Vegetables followed by Household.

# %%
# most sold item type
Train_data['Item_Type'].value_counts()

# %%
# Distribution of the outlet_Size
plt.figure(figsize=(10,8))
sns.countplot(Train_data['Outlet_Size'])
plt.show()

# %% [markdown]
# - most of the outlets across the country are Medium size.
# - very few outlet is High.

# %%
Train_data['Outlet_Size'].value_counts()

# %%
# Distribution of Outlet_Location_Type
plt.figure(figsize=(10,8))
sns.countplot(Train_data['Outlet_Location_Type'])
plt.show()

# %% [markdown]
# - Most of the products located in Tier 3.

# %%
Train_data['Outlet_Location_Type'].value_counts()

# %%
# Distribution of outlet types
plt.figure(figsize=(10,8))
sns.countplot(Train_data['Outlet_Type'])
plt.show()

# %%
Train_data['Outlet_Type'].value_counts()

# %%
# Relationship among item_weight and item_outlet_analysis
plt.figure(figsize=(13,9))
plt.xlabel('Item_Weight')
plt.ylabel('Item_Outlet_Sales')
plt.title('Item weight and Item Outlet Sales Analysis')
sns.scatterplot(x='Item_Weight', y='Item_Outlet_Sales', hue='Item_Type', size='Item_Weight', data=Train_data)
plt.show()

# %%
# Relationship between Item_Visibility and Item_Outlet_Sales
plt.figure(figsize=(13,9))
plt.xlabel('Item Visibility')
plt.ylabel('Item_Outlet_Sales')
plt.title('Item visibility and item outlet sales analysis')
sns.scatterplot(x='Item_Visibility', y='Item_Outlet_Sales', hue='Item_Type', size='Item_Weight', data=Train_data)
plt.show()

# %%
plt.figure(figsize=(12,7))
plt.xlabel('Item Visibility')
plt.ylabel('Maximum Retail Price')
plt.title('Item Visibility and Maximum Retail Price')
plt.plot(Train_data.Item_Visibility, Train_data.Item_MRP, ".", alpha=0.3, color='red')
plt.show()

# %%
# Distribution of Item_Fat_Content
plt.figure(figsize=(10,8))
sns.countplot(Train_data['Item_Fat_Content'])
plt.show()

# %%
Train_data['Item_Fat_Content'].value_counts()

# %%
# Rename the fat content column to readable
df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF' : 'Low Fat', 'reg' : 'Regular', 'low fat' : 'Low Fat'})

# %%
df['Item_Fat_Content'].value_counts()

# %%
# Doing this into train data
Train_data['Item_Fat_Content'] = Train_data['Item_Fat_Content'].replace({'LF' : 'Low Fat', 'reg' : 'Regular', 'low fat' : 'Low Fat'})

# %%
# Distribution of Item_Fat_Content
plt.figure(figsize=(10, 8))
sns.countplot(Train_data['Item_Fat_Content'])
plt.show()

# %%
Train_data['Item_Fat_Content'].value_counts()

# %%
# Check the correlation
Train_data.corr()

# %%
# Correlation heatmap
plt.figure(figsize=(15,15))
sns.heatmap(Train_data.corr(), annot=True, square=True, cmap='viridis', linecolor='k', linewidths=0.1)
plt.title('Correlation between different variables')
plt.show()

# %%
Train_data.columns


# %% [markdown]
# *Dealing with null values*

# %% [markdown]
# Item_Weight and Outlet_size columns having null values.

# %% [markdown]
# - Item_Weight has normally distributed and we can replace null value with mean

# %%
df['Item_Weight'].mean()

# %%
df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True)

# %% [markdown]
# - Outlet_size column is a object and replace with Medium size which is most shown value in that column.

# %%
df['Outlet_Size'].value_counts()

# %%
df['Outlet_Size'].fillna('Medium', inplace=True)

# %%
df.isnull().sum()

# %% [markdown]
# **Item Visibility**

# %%
Train_data.describe().T

# %% [markdown]
# - The Item_Visibility column minimum value is 0 and we need to treat that value as missing value. The Item 0 means it not available. Instead of NaN it marked as 0.

# %%
# count the total number of 0 in the Item Visibility column.
df[df['Item_Visibility'] == 0]['Item_Visibility'].count()

# %% [markdown]
# The Item Visibility column distribution is positive skewed and ideal representation is Median value to replace the missing value.

# %%
df['Item_Visibility'].fillna(df['Item_Visibility'].median(), inplace=True)

# %% [markdown]
# **Outlet Years**

# %%
df['Outlet_Establishment_Year'].value_counts()

# %% [markdown]
# to know how the period to making new outlet established from the starting(1985) and 2009.

# %%
df['Outlet_Years'] = 2009 - df['Outlet_Establishment_Year']
df['Outlet_Years'].describe().T

# %%
df['Item_Type'].value_counts()

# %%
Train_data.columns

# %% [markdown]
# The Item_Type associated with an Item_Identifier 

# %%
df['Item_Identifier'].value_counts()

# %% [markdown]
# group them into three category FD(Food), DR(Drink) and NC(Non consumable).

# %%
# pick only first two character eg: 'FDU15' take only index 0,1 which means (FD)
df["New_Item_Type"] = df["Item_Identifier"].apply(lambda x: x[0:2])

# %%
# Rename the values FD=Food, DR=Drink and NC= Non Consumable
df["New_Item_Type"] = df["New_Item_Type"].map({'FD':'Food', 'NC':'Non-Consumable', 'DR':'Drinks'})

# %%
df['New_Item_Type'].value_counts()

# %% [markdown]
# If a product is non consumable then why associate a fat content to them.Locate the non consumable in "New Item Type" column and Non consumable associate with Fat content and rename it as Non-edible.

# %%
df.loc[df['New_Item_Type'] == "Non-Consumable","Item_Fat_Content"] = "Non-Edible"

# %%
df['Item_Fat_Content'].value_counts()

# %%
item_visib_avg = df.pivot_table(values='Item_Visibility', index="Item_Identifier")

# %%
item_visib_avg

# %%
function = lambda x: x['Item_Visibility'] / item_visib_avg['Item_Visibility'][item_visib_avg.index == x["Item_Identifier"]][0]
df["item_visib_avg"] = df.apply(function, axis=1).astype(float)

# %%
df.head()

# %% [markdown]
# **Dealing with Categorical Varibales**

# %%
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()

df['Outlet'] = le.fit_transform(df['Outlet_Identifier'])
variable = ['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size', 'New_Item_Type', 'Outlet_Type', 'Outlet']
for i in variable:
    df[i] = le.fit_transform(df[i])

# %%
df.head()

# %% [markdown]
# create dummy variables for these Label Encoded variables to avoid our algorithm ranking these labels

# %%
# Dummy variable
df = pd.get_dummies(df, columns=['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size', 'New_Item_Type', 'Outlet_Type', 'Outlet'])
df.dtypes

# %%
df.head()

# %% [markdown]
# ### **Model Building**

# %% [markdown]
# We created the item type column to 3 category(food, non-consumable and drinks) we don't need that 16 category for that we need to drop that Item_Type column

# %%
df.drop(['Item_Type', 'Outlet_Establishment_Year'], axis=1, inplace=True)

# %%
# select and store the source train and test
train = df.loc[df['source'] == 'train']
test = df.loc[df['source'] == 'test']

# %%
# drop the source column in train data 
train.drop(['source'], axis=1, inplace=True)

# %%
# drop sales and test column in test data
test.drop(['Item_Outlet_Sales', 'source'], axis=1, inplace=True)

# %%
# drop the sales which our target column and stored into as y_train column and also drop Item_Identifier and Outlet_Identifier columns.
X_train = train.drop(['Item_Outlet_Sales', 'Item_Identifier', 'Outlet_Identifier'], axis=1)
y_train = train['Item_Outlet_Sales']
X_test = test.drop(['Item_Identifier', 'Outlet_Identifier'], axis=1).copy()

# %% [markdown]
# - ##  **Linear Regression**

# %%
lr = LinearRegression(normalize=True)
lr.fit(X_train, y_train)

# %%
lr_pred = lr.predict(X_test)

# %%
lr_pred

# %%
lr_accuracy = round(lr.score(X_train, y_train)*100)
lr_accuracy

# %% [markdown]
# - ## **DecisionTree Regressor**

# %%
dtree = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
dtree.fit(X_train, y_train)

# %%
dtree_pred = dtree.predict(X_test)
dtree_pred

# %%
dtree_accuracy = round(dtree.score(X_train, y_train)*100)
dtree_accuracy

# %% [markdown]
# ## **Random Forest Regressor**

# %%
from sklearn.model_selection import KFold, GridSearchCV

# %%
rfr = RandomForestRegressor(n_estimators=400, max_depth=6, min_samples_leaf=100, n_jobs=2)
rfr.fit(X_train, y_train)

# %%
rf_pred = rfr.predict(X_test)
rf_pred

# %%
rf_accuracy = round(rfr.score(X_train, y_train)*100)
rf_accuracy

# %%
# using hyper parameter tuning
param_grid = {'n_estimators': [200, 300, 400],
               'max_depth': [2, 5, 8], 'max_features': [2, 5, 8], 'min_samples_leaf': [10, 20, 30],
               'n_jobs': [2, 5, 8]}

# %%
gs = GridSearchCV(RandomForestRegressor(), param_grid=param_grid, cv=5)
gs.fit(X_train, y_train)

# %%
gs.best_params_

# %%
Rfr = RandomForestRegressor(n_estimators=300, max_depth=8, max_features=8, min_samples_leaf=10, n_jobs=2)
Rfr.fit(X_train, y_train)

# %%
Rfr_pred = Rfr.predict(X_test)
Rfr_pred

# %%
Rfr_accuracy = round(Rfr.score(X_train, y_train)*100)
Rfr_accuracy

# %% [markdown]
# ## **XGBoost Regressor**

# %%
from xgboost import XGBRegressor

xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05)
xgb.fit(X_train, y_train)

# %%
xgb_pred = xgb.predict(X_test)
xgb_pred

# %%
xgb_accuracy = round(xgb.score(X_train, y_train)*100)
xgb_accuracy

# %% [markdown]
# ## **Bagging Regressor**

# %%
from sklearn.ensemble import BaggingRegressor

bgr = BaggingRegressor(base_estimator=dtree, n_estimators=50, random_state=42)
bgr.fit(X_train, y_train)

# %%
bgr_pred = bgr.predict(X_test)
bgr_pred

# %%
bgr_accuracy = round(bgr.score(X_train, y_train)*100)
bgr_accuracy

# %% [markdown]
# ## **Support Vector Regressor**

# %%
from sklearn import svm 

svr = svm.SVR(C=5)
svr.fit(X_train, y_train)

# %%
svr_pred = svr.predict(X_test)
svr_pred

# %%
svr_accuracy = round(svr.score(X_train, y_train)*100)
svr_accuracy

# %% [markdown]
# ## **GradientBoost Regressor**

# %%
from sklearn.ensemble import GradientBoostingRegressor 

gbr = GradientBoostingRegressor(n_estimators=50, random_state=42)
gbr.fit(X_train, y_train)

# %%
gbr_pred = gbr.predict(X_test)
gbr_pred

# %%
gbr_accuracy = round(gbr.score(X_train, y_train)*100)
gbr_accuracy

# %%
# showing all algorithm accuracies
print('LinearRegression Accuracy:', lr_accuracy)
print('DecisionTreeRegressor Accuracy:', dtree_accuracy)
print('RandomForestRegressor Accuracy:', rf_accuracy)
print('RandomForestRegressor hyper Accuracy:', Rfr_accuracy)
print('XGBRegressor Accuracy:', xgb_accuracy)
print('SVR Accuracy:', svr_accuracy)
print('GradientBoostingRegressor Accuracy:', gbr_accuracy)

# %% [markdown]
# here i'm showing all the training accuracies and the better score is showing XGBoost Regressor 88 and the worst accuracy is showing support vector regressor the score is 30.


