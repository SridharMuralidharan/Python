# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 16:17:36 2018

@author: sridh
"""

from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


import pandas
import seaborn as sns
import numpy as np
sns.set_style('darkgrid')


csvfile='russiadata.newversion.csv'
adataframe=pandas.read_csv(csvfile)

adataframe1=adataframe[['Ad Text ','Ad Impressions ','Ad Clicks ','Ad Spend ']]


 	
#!pip install xgboost

from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x = adataframe.iloc[:,3:4]
y = adataframe.iloc[:,5:8]


# fit model no training data
#model = XGBClassifier()
#model.fit(x, y)

#!pip install textblob

from textblob import TextBlob
    
adataframe=adataframe.dropna()

x=adataframe['Ad Text ']
y=x.tolist()


strs = ['']*1879
strs1 = ['']*1879
strs2=['']*1879
 
#Polarity 
for i in range(0, len(y)-1):
   strs[i]=(y[i])
   strs1[i]=TextBlob(str(strs[i]))
   strs2[i]=strs1[i].sentiment.polarity
   
   
subs = ['']*1879
subs1= ['']*1879
subs2= ['']*1879
 
#Subjectivity 
for i in range(0, len(y)-1):
   subs[i]=(y[i])
   subs1[i]=TextBlob(str(strs[i]))
   subs2[i]=strs1[i].sentiment.subjectivity
   
   
adataframe1=adataframe[['Ad Text ','Ad Impressions ','Ad Clicks ','Ad Spend ']]   

se = pandas.Series(strs2)

#Then add the values to the DataFrame:


column_values = pandas.Series(strs2)
adataframe1.insert(loc=0, column='Polarity', value=column_values)


column_values = pandas.Series(subs2)
adataframe1.insert(loc=0, column='Subjectivity', value=column_values)

#!pip install vadersentiment1

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyze = SentimentIntensityAnalyzer()


pre = ['']*1879
pre1 = ['']*1879
pre2 = ['']*1879
pre3 = ['']*1879
pre4 = ['']*1879
 
#Subjectivity 
for i in range(0, len(y)-1):
   pre[i]=(y[i])
   pre1[i]=analyze.polarity_scores(str(pre[i]))['neg']
   pre2[i]=analyze.polarity_scores(str(pre[i]))['pos']
   pre3[i]=analyze.polarity_scores(str(pre[i]))['neu']
   pre4[i]=analyze.polarity_scores(str(pre[i]))['compound']


column_values = pandas.Series(pre1)
adataframe1.insert(loc=0, column='Negative', value=column_values)

column_values = pandas.Series(pre2)
adataframe1.insert(loc=0, column='Neutral', value=column_values)

column_values = pandas.Series(pre3)
adataframe1.insert(loc=0, column='Positive', value=column_values)

column_values = pandas.Series(pre4)
adataframe1.insert(loc=0, column='Compound-Sensitivity', value=column_values)


adataframe1.to_csv("Final_Prediction.csv", index=False, header=True)

