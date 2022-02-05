#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import nltk
from os import getcwd
import pandas as pd
from nltk.corpus import twitter_samples   
import matplotlib.pyplot as plt           
import random         
import re
import string                             
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer


# In[3]:


nltk.download('twitter_samples')
nltk.download('stopwords')


# In[10]:


all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')


# In[11]:


test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg 
test_x = test_pos + test_neg


# In[12]:


train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)


# In[13]:


from utils import process_tweet, build_freqs


# In[14]:


freqs = build_freqs(train_x, train_y)

print("type(freqs) = " + str(type(freqs)))
print("len(freqs) = " + str(len(freqs.keys())))


# In[15]:


def sigmoid(z): 
    '''
    Input:
        z: is the input (can be a scalar or an array)
    Output:
        h: the sigmoid of z
    '''

    h = 1/(1 + np.exp(-z))

    
    return h
print(sigmoid(-1.38))


# In[16]:


def gradientDescent(x, y, theta, alpha, num_iters):

 
    m = x.shape[0]
    
    for i in range(0, num_iters):
        
        z = np.dot(x,theta)
        
        h = sigmoid(z)
        
        J = (-1/m) * (np.dot(y.T,np.log(h)) + np.dot((1-y).T,(np.log(1-h))))

        theta = theta - ((alpha/m)*(np.dot(x.T,(h-y))))
        
    J = float(J)
    return J, theta


# In[25]:


def extract_features(tweet, freqs):
    '''
    Input: 
        tweet: a list of words for one tweet
        freqs: a dictionary corresponding to the frequencies of each tuple (word, label)
    Output: 
        x: a feature vector of dimension (1,3)
    '''
    word_l = process_tweet(tweet)
    
    x = np.zeros((1, 3)) 
    
    x[0,0] = 1 
    
    
    for word in word_l:
        
        x[0,1] += freqs.get((word, 1.0),0)
        
        x[0,2] += freqs.get((word,0),0)
        
    assert(x.shape == (1, 3))
    return x
print(extract_features(train_x[100],freqs))
print(extract_features(train_x[200],freqs))
print(extract_features(train_x[300],freqs))
print(extract_features(train_x[400],freqs))
print(extract_features(train_x[500],freqs))
print(extract_features(train_x[600],freqs))
print(extract_features(train_x[700],freqs))
print(extract_features(train_x[800],freqs))
print(extract_features(train_x[900],freqs))
print(extract_features(train_x[1000],freqs))


# In[18]:


def predict_tweet(tweet, freqs, theta):


    x = extract_features(tweet,freqs)
    
    y_pred = sigmoid(np.dot(x,theta))
    
    
    return y_pred


# In[19]:


def test_logistic_regression(test_x, test_y, freqs, theta):


    y_hat = []
    
    for tweet in test_x:
        y_pred = predict_tweet(tweet, freqs, theta)
        
        if y_pred > 0.5:
            y_hat.append(1)
        else:
            y_hat.append(0)

    accuracy = (y_hat==np.squeeze(test_y)).sum()/len(test_x)

    
    return accuracy


# In[36]:


X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i, :]= extract_features(train_x[i], freqs)

Y = train_y



J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)
print(f"The cost after training is {J:.8f}.")
print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta)]}")


# In[21]:


tmp_accuracy = test_logistic_regression(test_x, test_y, freqs, theta)
print(f"Logistic regression model's accuracy = {tmp_accuracy:.4f}")


# In[27]:


print('Label Predicted Tweet')
for x,y in zip(test_x,test_y):
    y_hat = predict_tweet(x, freqs, theta)
    print('THE TWEET IS:', x)
    print('THE PROCESSED TWEET IS:', process_tweet(x))
    print('%d\t%0.8f\t%s' % (y, y_hat, ' '.join(process_tweet(x)).encode('ascii', 'ignore')))


# In[21]:


freqs_anlz = {}
for keys in freqs:
    freqs_key = (keys)
    freqs_anlz[freqs_key] = np.random.uniform(-20,20)
print(freqs_anlz)


# In[17]:


w_anlz = 0
iteration = 1500
alpha = 1e-9
for i in range(iteration):
    for tweet in test_x:
        predicted = predict_tweet(tweet,freqs,theta)
        leng = len(process_tweet(tweet))
        if predicted > 0.5:
            for words in process_tweet(tweet):
                w_anlz = (freqs.get((words,1),0) *B +  L*(predicted - C*(x_hat.get((words,1),0)) + A * (x_hat)
                x_hat.get((words,1),0) = sigmoid(w_anlz)
                x_hat[words,1] = x_hat.get((words,1),0) -(alpha/(leng)*(w_anlz * (C * x-hat - 1)))
                
        else:
            for wordz in process_tweet(tweet):
                w_anlz = (freqs.get((words,0),0) *B +  L*(predicted - C*(x_hat.get((words,0),0)) + A * (x_hat)
                x_hat.get((words,0),0) = sigmoid(w_anlz)
                x_hat[words,0] = x_hat.get((words,0),0) -(alpha/(leng)*(w_anlz * (C * x-hat - 0)))
        


# In[18]:


print (x_hat)


# In[28]:





# In[33]:


print(train_x[100])
print(process_tweet(train_x[100]))


# In[52]:


J1, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1)
J2, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 100)
J3, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 200)
J4, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 300)
J5, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 400)
J6, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 500)
J7, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 600)
J8, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 700)
J9, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 800)
J10, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 900)
J11, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1000)
J12, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1200)
J13, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)
fig, ax = plt.subplots(figsize = (5, 5))

x =   (1,100,200,300,400,500,600,700,800,900,1000,1200,1500)

y=  (J1,J2,J3,J4,J5,J6,J7,J8,J9,J10,J11,J12,J13)

ax.scatter(x, y)  


plt.xlabel("iteration")
plt.ylabel("cost")


# In[ ]:





# In[ ]:




