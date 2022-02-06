# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 18:25:49 2022

@author: wendy
"""
'''
Sources: 
    Ken Jee Kaggle Titanic Project
    Analyticsvidhya.com
'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Importing Data

train=pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#Reading Data
train['train_test'] = 1
test['train_test'] = 0
test['Survived'] = np.NaN
all_data = pd.concat([train,test])

#Data Exploration
train.info()
train.describe()

#Separating data into Numeric & Categorical Attributes
df_num = train[['Age', 'SibSp', 'Parch', 'Fare']]
df_cat = train[['Survived', 'Pclass', 'Sex', 'Ticket', 
                'Cabin', 'Embarked']]

#Numerical
for i in df_num.columns:
    plt.hist(df_num[i])
    plt.title(i)
    plt.show()

'''
Ages looks almost normal, skews somewhat to the right.
The others look scattered
'''

sns.heatmap(df_num.corr())
'''
Parch & SibSp appear to have a higher correlation - Parents are more likely to be traveling with multiple children
And spouses would likely travel together
'''
#Compare Survival across Numerical Attributes
pd.pivot_table(train, index='Survived', values = ['Age','SibSp', 'Parch', 'Fare'])
'''
Age: Younger persons more likely to survive
Fare: Persons who paid higher fare were more likely to survive (Probably first-class)
Parch: More likely to survive if travelling with parents ( Parents likely protect/save kids)
SibSp: Less likely to survive if travelling with siblings
'''

#Categorical
for i in df_cat.columns:
    sns.barplot(df_cat[i].value_counts().index, df_cat[i].value_counts()).set_title(i)
    plt.show()
    
'''
Survived: Most persons died
Pclass: Most persons were travelling on 3rd class tickets
Sex: More males than females
Embarked: Most persons boarded at Southampton
Ticket & Cabin: not much info can be taken from graphs (inputs are a mix of numbers and letters - no inference can be made)
'''
#Comparing Survival to Categorical Attributes
print(pd.pivot_table(train, index='Survived', columns = "Pclass",
                     values = 'Ticket', aggfunc='count'))
print()
print(pd.pivot_table(train, index='Survived', columns ='Sex',
                     values='Ticket', aggfunc ='count'))
print()
print(pd.pivot_table(train, index='Survived',columns ='Embarked',
                     values='Ticket', aggfunc='count'))
print()
'''
Pclass: 63% of 1st class survived, 24% of 2nd class survived & 47% of 3rd class survived
    - the rich survived
Sex: More females survived (74% of women survived, 19% of men survived)
    - "Women & children first"
Embarked: Cherbourg- 55% survived, Queenstown - 39% survived, Southampton - 34% survived
'''
#FEATURE ENGINEERING
''' 
Cabin - Simplify cabins
    -did cabin letter or having multiple cabins impact survival
Tickets - Did the types of tickets affect survival
Does a persons title affect their survival
'''
#Split Cabin numbers into Cabin Numbers
df_cat.Cabin
train['cabin_multi'] = train.Cabin.apply(lambda x: 0 if pd.isna(x)
                                            else len(x.split(' ')))
train['cabin_multi'].value_counts()
'''
Most persons shared cabins and very few had multiple cabins
'''
pd.pivot_table(train, index='Survived', columns='cabin_multi',
               values='Ticket', aggfunc='count')
# n = null
# Treating null values like its own category
# Split Cabin into cabin letters/levels
train['cabin_adv'] = train.Cabin.apply(lambda x: str(x)[0])
print(train.cabin_adv.value_counts())
pd.pivot_table(train, index='Survived', columns='cabin_adv',
               values='Name', aggfunc='count')

#Splitting Ticket values
train['ticket_num']=train.Ticket.apply(lambda x:1 if x.isnumeric() else 0)
train['ticket_let']=train.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1])
                                       .replace('.','').replace('/','')
                                       .lower() if len(x.split(' ')[:-1])>0 else 0)
#Title of Passengers
train.Name.head(50)
train['name_title']= train.Name.apply(lambda x: x.split(',')[1]
                                      .split('.')[0].strip())
train['name_title'].value_counts()

#DATA PREPROCESSING
'''
-Drop null values
-Remove unnecessary data
-Categorically transform all data
Impute data with central tendencies for age & fare
Normalize fare
Scaled data 0-1 with standard scaler
'''
# Input Nulls - Using Median
all_data['cabin_multi'] = all_data.Cabin.apply(lambda x: 0 if pd.isna(x) else len(x.split(' ')))
all_data['cabin_adv'] = all_data.Cabin.apply(lambda x: str(x)[0])
all_data['ticket_num'] = all_data.Ticket.apply(lambda x: 1 if x.isnumeric() else 0)
all_data['ticket_let'] = all_data.Ticket.apply(lambda x: ''.join(x.split(' ')[:-1]).replace('.','').replace('/','').lower() if len(x.split(' ')[:-1]) >0 else 0)
all_data['name_title'] = all_data.Name.apply(lambda x: x.split(',')[1].split('.')[0].strip())
all_data.Age= all_data.Age.fillna(train.Age.median())
all_data.Fare= all_data.Fare.fillna(train.Fare.median())

# Drop null 'Embarked rows
all_data.dropna(subset=['Embarked'], inplace=True)

#Log-Norm of Fare
all_data['norm_fare']=np.log(all_data.Fare+1)
all_data['norm_fare'].hist

#Converted fare to category for pd.get_dummies()
all_data.Pclass=all_data.Pclass.astype(str)

#Create dummyvariables from categories
all_dummies= pd.get_dummies(all_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'norm_fare',
                                      'Embarked', 'cabin_adv', 'cabin_multi', 'ticket_num', 'name_title', 'train_test']])
#Split to train test 
X_train= all_dummies[all_dummies.train_test == 1].drop(['train_test'], axis= 1)
X_test= all_dummies[all_dummies.train_test == 0].drop(['train_test'], axis =1)

y_train = all_data[all_data.train_test==1].Survived 
y_train.shape

#Scaling data
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
all_dummies_scaled = all_dummies.copy()
all_dummies_scaled[['Age','SibSp','Parch','norm_fare']]= scale.fit_transform(all_dummies_scaled[['Age','SibSp','Parch','norm_fare']])
all_dummies_scaled

X_train_scaled = all_dummies_scaled[all_dummies_scaled.train_test == 1].drop(['train_test'], axis =1)
X_test_scaled = all_dummies_scaled[all_dummies_scaled.train_test == 0].drop(['train_test'], axis =1)

y_train = all_data[all_data.train_test==1].Survived

#MODEL BUILDING
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
#Naive Bayes 
gnb = GaussianNB()
cv = cross_val_score(gnb,X_train_scaled,y_train,cv=5)
print(cv)
print(cv.mean())

#Logisitic Regression
lr =LogisticRegression(max_iter=2000)
cv=cross_val_score(lr,X_train_scaled,y_train,cv=5)
print(cv)
print(cv.mean())

#Decision Tree
dt= tree.DecisionTreeClassifier(random_state=1)
cv=cross_val_score(dt, X_train_scaled,y_train,cv=5)
print(cv)
print(cv.mean())

#K Means CLustering
knn=KNeighborsClassifier()
cv=cross_val_score(knn,X_train_scaled,y_train,cv=5)
print(cv)
print(cv.mean())

#Random Forest
rf=RandomForestClassifier(random_state=1)
cv=cross_val_score(rf,X_train_scaled,y_train,cv=5)
print(cv)
print(cv.mean())

#Support Vector Classifier
svc= SVC(probability=True)
cv=cross_val_score(svc,X_train_scaled,y_train,cv=5)
print(cv)
print(cv.mean())


