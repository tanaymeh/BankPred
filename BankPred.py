# Bank Prediction Model v0.1 using Keras with Tensorflow backend
# Author: Tanay Mehta
# Github: http://github.com/tanaymehta28/BankPred
# Licensed under MIT License

# Import all the modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler

# Read the CSV file named as 'modelling.csv'
data = pd.read_csv('modelling.csv')

# Clean the array up by listing only the features we need
X = data.iloc[:, 3:13].values
Y = data.iloc[:, 13].values

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# First encode the non-integer labels like Country and Gender and then fit them in
LabelEncoder_X_1 = LabelEncoder()
X[:,1] = LabelEncoder_X_1.fit_transform(X[:,1])
LabelEncoder_X_2 = LabelEncoder()
X[:,2] = LabelEncoder_X_2.fit_transform(X[:,2])

# Also, Encode all the Integral values using OHE
OHE = OneHotEncoder(categorical_features=[1])
X = OHE.fit_transform(X).toarray()
X = X[:,1:]

from sklearn.model_selection import train_test_split

#Split the data into training Data and testing data
trainX,testX,trainY,testY = train_test_split(X,Y,test_size=0.2,random_state=0)

#Transform the training batch of X and testing batch of X
sc = StandardScaler()
trainX = sc.fit_transform(trainX)
testX = sc.transform(testX)

'''Boring Data Preprocessing & Cleaning is complete here. 
Now comes the real awesome part: Making and Training the ANN!!'''

import keras
from keras.layers import Dense
from keras.models import Sequential

# Model initialization
model = Sequential()
# This model has two hidden layers with 6 neurons each with Rectified Linear Unit activation
inputLayer = Dense(units=6,kernel_initializer='uniform',activation='relu',input_dim=11)
hiddenLayer1 = Dense(units=6,kernel_initializer='uniform',activation='relu')
hiddenLayer2 = Dense(units=6,kernel_initializer='uniform',activation='relu')
outputLayer = Dense(units=1,kernel_initializer='uniform',activation='sigmoid')

# Add the layers to the model and compile the model
model.add(inputLayer)
model.add(hiddenLayer1)
model.add(hiddenLayer2)
model.add(outputLayer)
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# fit the model with training batch of X and testing batch of X
# If you increase the batch size; the training process will end fast, but it 'might' decrease the accuracy
# Also, 100 epochs is 'good enough', any more than that will be overkill!

model.fit(trainX,trainY,batch_size=10,epochs=100)

# model.save('bankPred')
# Now predict on the testing batch of X
predY = model.predict(testX)

# display(predY)

# Note: I have set the %age of prediction to be more than 70%, if you want to make your model predictions to be more flexible, alter the values.
# This last 'predY' means that if any person has more than 70% chance of leaving the bank, they will be listed as 'True' in the 'predY' variable.
predY = (predY>0.7)
# display(predY)

# Use the below one only if you want to view the whole array (for users using Jupyter notebooks or azure notebooks)
# If you are using Spider IDE; just comment it out and view any variable in 'Variable Explorer' 
np.set_printoptions(threshold=np.nan)
display(predY)

# Use the confusion matrix to get the number of correct predictions out of 2000
# The value of 'cm' will be something like, 
# cm = ([1580, 15]
#       [262, 43])
# In this Confusion Matrix, the correct predicted values are the top left one + bottom right one,
# which in this case is 1580 + 43 = 1723
# And the accuracy will be == correct prediction/2000 

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(testY,predY)

'''Any new prediction can be evaluated by using the ".predict()" method with proper parameters"'''