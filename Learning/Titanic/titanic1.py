import pandas as pd
import numpy as np
import random as rnd
import os
import re
import itertools
import scipy.stats as st

# Write model submissions?
save = True
n_row = None
debug = False
if debug is True:
    n_row = 80

# Master Parameters:
n_splits = 2 # Cross Validation Splits
n_iter = 70 # Randomized Search Iterations
scoring = 'accuracy' # Model Selection during Cross-Validation
rstate = 27 # Random State used
testset_size = 0.35

# Boosting rounds
num_rounds = 800

# Trees Parameters
n_tree_range = st.randint(100, num_rounds)

# Load
train_df = pd.read_csv("./train.csv", index_col='PassengerId', nrows=n_row)
test_df = pd.read_csv("./test.csv", index_col='PassengerId', nrows=n_row)
# train_df = pd.read_csv("Titanic Support/train.csv", index_col='PassengerId')
# test_df = pd.read_csv("Titanic Support/test.csv", index_col='PassengerId')

# For Pre-Processing, combine train/test to simultaneously apply transformations
Survived = train_df['Survived'].copy()
train_df = train_df.drop('Survived', axis=1).copy()
df = pd.concat([test_df, train_df])
traindex = train_df.index
testdex = test_df.index
del train_df
del test_df

# New Variables engineering, heavily influenced by:
# Kaggle Source- https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python
# Family Size
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
# Name Length
df['Name_length'] = df['Name'].apply(len)
# Is Alone?
df['IsAlone'] = 0
df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

# Title: (Source)
# Kaggle Source- https://www.kaggle.com/ash316/eda-to-prediction-dietanic
df['Title']=0
df['Title']=df.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations
df['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],
                    ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)

# Age
df.loc[(df.Age.isnull())&(df.Title=='Mr'),'Age']= df.Age[df.Title=="Mr"].mean()
df.loc[(df.Age.isnull())&(df.Title=='Mrs'),'Age']= df.Age[df.Title=="Mrs"].mean()
df.loc[(df.Age.isnull())&(df.Title=='Master'),'Age']= df.Age[df.Title=="Master"].mean()
df.loc[(df.Age.isnull())&(df.Title=='Miss'),'Age']= df.Age[df.Title=="Miss"].mean()
df.loc[(df.Age.isnull())&(df.Title=='Other'),'Age']= df.Age[df.Title=="Other"].mean()
df = df.drop('Name', axis=1)

# Fill NA
# Categoricals Variable
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode().iloc[0])

# Continuous Variable
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

## Assign Binary to Sex str
df['Sex'] = df['Sex'].map({'female': 1,
                           'male': 0}).astype(int)
# Title
df['Title'] = df['Title'].map( {'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master':3, 'Other':4} )#.astype(int)
df['Title'] = df['Title'].fillna(df['Title'].mode().iloc[0])
df['Title'] = df['Title'].astype(int)

# Embarked
df['Embarked'] = df['Embarked'].map( {'Q': 0, 'S': 1, 'C': 2} ).astype(int)

# Get Rid of Ticket and Cabin Variable
df= df.drop(['Ticket', 'Cabin'], axis=1)

categorical_features = ["Pclass","Sex","IsAlone","Title", "Embarked"]

print(df.head())