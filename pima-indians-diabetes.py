#!/usr/bin/env python
# coding: utf-8

# In[36]:


# first neural network with keras tutorial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense


# In[37]:


# load the dataset
rawdata = pd.read_csv('pima-indians-diabetes.csv')
dataset = np.array (rawdata)
# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]


# In[38]:


# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[39]:


# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[40]:


# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10)


# In[41]:


# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))


# In[42]:


# make class predictions with the model  
#predictions = (model.predict(X) > 0.5).astype(int)
# make probability predictions with the model v2
predictions = model.predict(X)
# round predictions 
rounded = [round(x[0]) for x in predictions]
for i in range(5):
    print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))


# In[ ]:




