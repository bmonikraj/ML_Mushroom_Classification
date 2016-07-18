import os
import time
import datetime
from collections import Counter
import pandas
from sklearn import preprocessing, naive_bayes
import numpy

t = datetime.datetime.now().second

dataFrame = pandas.read_csv(str(os.getcwd())+"\\agaricus-lepiota.csv")
#Reading file done

for i in dataFrame:
    missingVal = Counter(dataFrame[str(i)]).most_common(1)[0][0]
    for j in range(0,len(dataFrame[str(i)]),1):
        if dataFrame.at[j,str(i)]=='?':
            dataFrame.set_value(j,str(i),str(missingVal))
#Categorical Imputation done based on mode (Most frequent value assigned to missing value)

y = dataFrame['A']
del dataFrame['A']
x = dataFrame
#Assigned X and Y

lableEncoderListX = [] #list  of LabelEncoder() objects for each column of X
for i in x:
    le = preprocessing.LabelEncoder()
    le.fit(x[str(i)])
    x[str(i)] = le.transform(x[str(i)])
    lableEncoderListX.append(le)

lableEncoderListY = [] #list  of LabelEncoder() objects for each column of Y
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)
lableEncoderListY.append(le)
#one to m encoding done for converting categorical variables to numeric for algorithms to run

multinomialNB = naive_bayes.MultinomialNB(class_prior=None, fit_prior=False)
multinomialNB.fit(x,y)

testY = numpy.array(multinomialNB.predict(x))
trainY = numpy.array(y)
ErrorY = numpy.array(numpy.subtract(testY,trainY))

error = 0
for k in range(0,ErrorY.size,1):
    if ErrorY[k]!=0:
        error=error+1
sizeOfErrorY = ErrorY.size
error = float(float(error)/float(sizeOfErrorY))
error = error*float(100)
print "Multinomial NB Percentage of error on training set is "+str(error)
#Multinomial Naive Bayes Classification

guassianNB = naive_bayes.GaussianNB()
guassianNB.fit(x,y)

testY = numpy.array(guassianNB.predict(x))
trainY = numpy.array(y)
ErrorY = numpy.array(numpy.subtract(testY,trainY))

error = 0
for k in range(0,ErrorY.size,1):
    if ErrorY[k]!=0:
        error=error+1
sizeOfErrorY = ErrorY.size
error = float(float(error)/float(sizeOfErrorY))
error = error*float(100)
print "Guassian NB Percentage of error on training set is "+str(error)
#Guassian Naive Bayes Classification

bernoulliNB = naive_bayes.BernoulliNB(class_prior=None,fit_prior=False)
bernoulliNB.fit(x,y)

testY = numpy.array(bernoulliNB.predict(x))
trainY = numpy.array(y)
ErrorY = numpy.array(numpy.subtract(testY,trainY))

error = 0
for k in range(0,ErrorY.size,1):
    if ErrorY[k]!=0:
        error=error+1
sizeOfErrorY = ErrorY.size
error = float(float(error)/float(sizeOfErrorY))
error = error*float(100)
print "Bernoulli NB Percentage of error on training set is "+str(error)
#Bernoulli Naive Bayes Classification

print "Time taken for analysis (sec) : "+str(datetime.datetime.now().second-t)

print "Done..."