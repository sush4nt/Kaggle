# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 20:22:02 2021

@author: sushantpatil

Link to the thread : https://www.kaggle.com/startupsci/titanic-data-science-solutions
"""

import numpy as np
import pandas as pd
import random as rnd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# loading the data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# looking at the data shape and its features
print(train.head(5))
print(train.shape)
print(test.shape)
print(train.columns)

# take a look at each column and split features as categorical and numerical
# Some of the categorical variables are : survived, embarked, sex  Ordinal : pclass

# some of the numerical variable are continuous: fare,age  discrete : parch, sibsp,

# Now, take a look at the values we are missing and data types of our features
# print('_'*30+ '\ntrain null vals')
# print(train.isnull().sum().sort_values(ascending=False))
# print('_'*30 + '\ntest null vals')
# print(test.isnull().sum().sort_values(ascending=False)) 
print(train.info())
#print(test.info())

# From the actual stats, around 40% of the people on board survived
# Now, we check the % survived on our given dataset
print(train.Survived.value_counts())
# Represents the actual survival stat data approx.
# Now we look at some categorical data and try to check if some categories are more inclined
print(train.Pclass.value_counts())
print(train.Sex.value_counts())
print(train.SibSp.value_counts())
print(train.Parch.value_counts())

# takeaways :
# there more lower (3rd) class people on the cruise
# more males
# most of the people had no sibling or spouse
# most of the had no parent or children

# Lets look at some distinct feature correlations
# Upper class have higher survival rates
print(train[['Pclass','Survived']].groupby(['Pclass']).mean())

# Females have higher survival rates
print(train[['Sex','Survived']].groupby(['Sex']).mean())


print(train[['SibSp','Survived']].groupby(['SibSp']).mean())
print(train[['Parch','Survived']].groupby(['Parch']).mean())

# Analysing by visualizing data

# # Correlating numerical features
# p1 = sns.FacetGrid( train, col='Survived' )
# p1.map( plt.hist, 'Age', bins = 20, edgecolor='black', color= '#DC143C', alpha= 0.7 )

# # Correlating numerical and ordinal features(Pclass)

# # We can combine multiple features for identifying correlations using a single plot.
# # This can be done with numerical and categorical features which have numeric values.
# p2 = sns.FacetGrid( train, col='Survived', row='Pclass', height=3, aspect= 1 )
# p2.map(plt.hist, 'Age', bins = 20, edgecolor='black', color= '#ADFF2F', alpha = 0.7 )

# # Correlating categorical features
# p3 = sns.FacetGrid( train , row='Embarked', height=3, aspect=2 )
# p3.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='pastel')
# p3.add_legend()

# # Something for me to ponder upon, the above line code just helped compare 
# # survival rates of male and female of each Pclass according to their embarkation
# # THATS 4 CATEGORICAL FEATS ONTO A 2D PLOT   

# # Correlating categorical and numerical features 
# p4 = sns.FacetGrid( train, row='Embarked', col='Survived', height=3, aspect=2 )
# p4.map( sns.barplot, 'Sex', 'Fare', alpha=0.6, ci= None)

# Correcting by dropping features

# we will be dropping Ticket and Cabin columns from the dataset 

train.drop(['Ticket','Cabin'], axis=1, inplace=True)
test.drop(['Ticket', 'Cabin'], axis=1, inplace=True)

print(train.shape)
print(test.shape)

# Feature Extraction/ Feature Engineering

# there's a brief passage extracting useful information from Name and Passenger Id
# for simplicity purpose, I am omitting the particular part of study and directly drop both those columns

train.drop(['Name', 'PassengerId'], axis=1, inplace=True)
test.drop(['Name'], axis=1, inplace=True)

print(train.shape)
print(test.shape)

#print(train.head(5))
from sklearn.preprocessing import LabelEncoder

lbl = LabelEncoder()
lbl.fit(train['Sex'])
# 1 is male
# 0 is female
train['Sex'] = lbl.fit_transform(train['Sex'])
test['Sex'] = lbl.fit_transform(test['Sex'])
lbl2= LabelEncoder()
lbl2.fit(train['Pclass'])
# 0 is upper 1 is middle 2 is lower class
train['Pclass'] = lbl2.fit_transform(train['Pclass'])
test['Pclass'] = lbl2.fit_transform(test['Pclass'])

#print(train.head(5))
#print(test.head(5))


# Now, lets deal with missing values for each feature one by one

#print(train.isnull().sum())
#print(test.isnull().sum())
# Age 
print(train['Age'].isnull().sum())
# We can fill NaN values with Median values
ageMed = np.nanmedian(train['Age'])
ageMed2 = np.nanmedian(test['Age'])
print(np.nanmedian(train['Age']))
print(np.nanmedian(test['Age']))
train['Age'].fillna(ageMed, inplace=True)
test['Age'].fillna(ageMed2, inplace=True)
print(train['Age'].isnull().sum())
print(test['Age'].isnull().sum())

train.loc[train['Age']<=16,'Age']=0
train.loc[(train['Age']>16) & (train['Age']<=32),'Age']=1
train.loc[(train['Age']>32) & (train['Age']<=64),'Age']=2
train.loc[(train['Age']>64),'Age']=3
test.loc[test['Age']<=16,'Age']=0
test.loc[(test['Age']>16) & (test['Age']<=32),'Age']=1
test.loc[(test['Age']>32) & (test['Age']<=64),'Age']=2
test.loc[(test['Age']>64),'Age']=3

#print(train.head(5))

# Creating new features combining existing features

train['FamilySize'] = train['Parch']+train['SibSp']+1
train['isAlone']=0
train.loc[train['FamilySize']==1, 'isAlone']=1

test['FamilySize'] = test['Parch']+test['SibSp']+1
test['isAlone']=0
test.loc[test['FamilySize']==1, 'isAlone']=1
#print(train.head(5))

# Now we can drop SibSp, Parch, FamilySize
train.drop(['SibSp','Parch','FamilySize'], axis=1, inplace=True)
test.drop(['SibSp','Parch','FamilySize'], axis=1, inplace=True)
#print(train.head(5))

train['AgeClass'] = train['Pclass']*train['Age']
test['AgeClass'] = test['Pclass']*test['Age']
print(train.head(5))
print(test.head(5))
print('-'*60)
#print(train.isnull().sum())
#print(test.isnull().sum())
embarkedMed = train['Embarked'].mode()
embarkedMed2 = test['Embarked'].mode()
#print(embarkedMed[0])
train['Embarked'].fillna(value=embarkedMed[0], inplace=True)
test['Embarked'].fillna(value=embarkedMed2[0], inplace=True)
#print(train.head(5))
fareMed = test['Fare'].median()
test['Fare'].fillna(fareMed, inplace=True)
train['Fare'] = train['Fare'].round(2)
test['Fare'] = test['Fare'].round(2)

#print(pd.qcut(train['Fare'], 4))
train.loc[train['Fare']<=7.91,'Fare']=0
train.loc[(train['Fare']>7.91) & (train['Fare']<=14.45),'Fare']=1
train.loc[(train['Fare']>14.45) & (train['Fare']<=31),'Fare']=2
train.loc[(train['Fare']>31),'Fare']=3
test.loc[test['Fare']<=7.91,'Fare']=0
test.loc[(test['Fare']>7.91) & (test['Fare']<=14.45),'Fare']=1
test.loc[(test['Fare']>14.45) & (test['Fare']<=31),'Fare']=2
test.loc[(test['Fare']>31),'Fare']=3
print(train.isnull().sum())
print(test.isnull().sum())

lbl3 = LabelEncoder()
lbl3.fit(train['Embarked'])
# C is 0 Q is 1 S is 2
train['Embarked'] = lbl3.fit_transform(train['Embarked'])
test['Embarked'] = lbl3.fit_transform(test['Embarked'])
print(train.head(5))
print(test.head(5))
#print(train.isnull().sum())
train['Fare'] = train['Fare'].astype(int)
train['Age'] = train['Age'].astype(int)
train['AgeClass'] = train['AgeClass'].astype(int)
test['Fare'] = test['Fare'].astype(int)
test['Age'] = test['Age'].astype(int)
test['AgeClass'] = test['AgeClass'].astype(int)
print('-'*60)
print(train.head(5))
print(test.head(5))
# Model Selection
y = train['Survived']
X = train.drop(['Survived'], axis=1)
X_test = test.drop(['PassengerId'], axis=1)

print(y.shape)
print(X.shape)
print(X_test.shape)
# print(y.head(5))
# print(X.head(5))
# print(X_test.head(5))


# LogisticRegression

logR = LogisticRegression()
logR.fit(X, y)
preds_log = logR.predict(X_test)
acc_log = logR.score(X, y)
#print(acc_log) 
coeff_df = pd.DataFrame(train.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logR.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)
#print(coeff_df.head(10))


# Support Vector Machines


svc = SVC()
svc.fit(X,y)
preds_svc = svc.predict(X_test)
acc_svc = svc.score(X,y)
#print(acc_svc)

# K Nearest Neighbors 


knn = KNeighborsClassifier( n_neighbors=6 )
knn.fit(X,y)
preds_knn = knn.predict(X_test)
acc_knn = knn.score(X,y)
#print(acc_knn)

# Gaussian Naive Bayes Classifier


gaussiannb = GaussianNB()
gaussiannb.fit(X,y)
preds_gnb = gaussiannb.predict(X_test)
acc_gnb = gaussiannb.score(X,y)
#print(acc_gnb) 

# Perceptron


perc = Perceptron()
perc.fit(X,y)
preds_perc = perc.predict(X_test)
acc_perc = perc.score(X,y)
#print(acc_perc)

# Linear SVC


linsvc = LinearSVC()
linsvc.fit(X,y)
preds_linsvc = linsvc.predict(X_test)
acc_linsvc = linsvc.score(X,y)
#print(acc_linsvc)

# Stochastic Gradient Descent


sgd = SGDClassifier()
sgd.fit(X,y)
preds_sgd = sgd.predict(X_test)
acc_sgd = sgd.score(X,y)
#print(acc_sgd)


# Decision Tree Classifier


dtree = DecisionTreeClassifier()
dtree.fit(X,y)
preds_dtree = dtree.predict(X_test)
acc_dtree = dtree.score(X,y)
#print(acc_dtree)

# Random Forest Classifier

rforest = RandomForestClassifier()
rforest.fit(X,y)
preds_rforest = rforest.predict(X_test)
acc_rforest = rforest.score(X,y)
#print(acc_rforest)

models = pd.DataFrame( {
    'Model' : [ 'Logistic Regression', 'SVM', 'KNN', 'GaussianNB', 'Perceptron', 'LinearSVC', 'SGD', 'DecisionTree', 'RandomForest' ],
    'Scores' : [ acc_log, acc_svc, acc_knn, acc_gnb, acc_perc, acc_linsvc, acc_sgd, acc_dtree, acc_rforest ]
    } )
print(models.head(10))
print('Run Complete!')

