
# coding: utf-8

# In[50]:


# EDA analysis of titanic dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

#Loading train  and test dataset and observing the dataset
train = pd.read_csv("trainb.csv")
train.head()
# setting passenger_id as index
train=train.set_index("PassengerId")
#Finding number rows and columns in the dataset
train.shape
test=pd.read_csv("testa.csv")
test.head()
# setting passenger_id as index
test=test.set_index("PassengerId")
#Finding number rows and columns in the test dataset
test
#Finding data types of the column
data_types=pd.DataFrame(train.dtypes)
#Renaming column 0
data_types=data_types.rename(columns={0:"Datatype"})
#Counting number of data columnwise
data_types["NoOfRows"]=train.count()
data_types["Missing_Values"]=train.isnull().sum()
data_types["Number_Unique"]=train.nunique()
#Analyzing Age attribute
train["Age"].describe()
#Analyzing categorical attribute
train.describe(include=["object"])
#univariate analysis
fig, axes = plt.subplots(2, 4, figsize=(16, 10))
sns.countplot('Survived',data=train,ax=axes[0,0])
sns.countplot('Pclass',data=train,ax=axes[0,1])
sns.countplot('Sex',data=train,ax=axes[0,2])
sns.countplot('SibSp',data=train,ax=axes[0,3])
sns.countplot('Parch',data=train,ax=axes[1,0])
sns.countplot('Embarked',data=train,ax=axes[1,1])
sns.distplot(train['Fare'], kde=True,ax=axes[1,2])
sns.distplot(train['Age'].dropna(),kde=True,ax=axes[1,3])
#bivariate analysis
figbi, axesbi = plt.subplots(2, 4, figsize=(16, 10))
train.groupby('Pclass')['Survived'].mean().plot(kind='barh',ax=axesbi[0,0],xlim=[0,1])
train.groupby('SibSp')['Survived'].mean().plot(kind='barh',ax=axesbi[0,1],xlim=[0,1])
train.groupby('Parch')['Survived'].mean().plot(kind='barh',ax=axesbi[0,2],xlim=[0,1])
train.groupby('Sex')['Survived'].mean().plot(kind='barh',ax=axesbi[0,3],xlim=[0,1])
train.groupby('Embarked')['Survived'].mean().plot(kind='barh',ax=axesbi[1,0],xlim=[0,1])
sns.boxplot(x="Survived", y="Age", data=train,ax=axesbi[1,1])
sns.boxplot(x="Survived", y="Fare", data=train,ax=axesbi[1,2])




# In[41]:


train.head()

