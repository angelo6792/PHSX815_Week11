#! /usr/bin/env python

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
# from tensorflow.keras.optimizers import SGD, Adam
# from tensorflow.keras.activations import relu
import tensorflow_datasets as tfds
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

#if you want to load the data as tuples, you can use the as_supervised=True argument
data_train = tfds.load('wine_quality',split='train',as_supervised=True)

#looking at just one sample of our data
pt = data_train.take(1)
# type(pt)

#can convert this TakeDataset object to a numpy array (can do this for the whole dataset too)
print("Features and label for first entry")
for features, label in tfds.as_numpy(pt):
    print(features,label)

#we want to load dataset as a a dictionary of tf.Tensors (can't transform tuples to dataframe)
data_train_white = tfds.load('wine_quality/white',split='train')
data_train_red = tfds.load('wine_quality/red',split='train')

#transform dictionary to dataframe - combining red and white wine
df_white = tfds.as_dataframe(data_train_white)
df_red = tfds.as_dataframe(data_train_red)
df = pd.concat([df_white,df_red])

print('number of samples',len(df['quality']))

#what are our output possibilities?
print('possible wine quality ratings',df['quality'].unique())


#do we have any missing data (empty or NaN entries in features or labels)?
dataNans = df.isnull().values.any()
if not dataNans:
    print("all good!")

#it's helpful to separate our input features from our target features (quality) 
#so we can later only transform our inputs without changing our labels
labels = df['quality']
df = df.drop(labels='quality',axis=1)
labels.unique()

enc = OneHotEncoder(sparse=False)
labels = enc.fit_transform(labels.to_numpy().reshape(-1,1))

#make our test data
df, df_test, labels, df_testLabels = train_test_split(df,labels,test_size=0.1)

#look at the first 5 entries
df.head()

df.describe()

#visualizing our input features
nFeatures = len(df.columns)
nCols = 3
nRows = int(np.ceil(nFeatures/nCols))
cols = df.columns
fig, axs = plt.subplots(nRows,nCols,figsize=(15,20))
# for i, ax in enumerate(axs)
col = 0
for i in range(nRows):
    for j in range(nCols):
        if col >= nFeatures:
            break
        h = axs[i,j].hist(df[cols[col]])
        h = axs[i,j].set_title(cols[col])
        col += 1

plt.scatter(df['features/free sulfur dioxide'],df['features/total sulfur dioxide'])

cols = df.columns
nClasses = len(labels[0])

#using Keras's Sequential model - https://keras.io/api/models/sequential/
model = Sequential()
#add input layer
model.add(Input(shape=(len(cols),))) #the input layer shape should match the number of features we have
#add first layer of fully connected neurons
model.add(Dense(64,activation='relu'))
#add second layer (first hidden layer)
model.add(Dense(64,activation='relu'))
#and one more because why not
model.add(Dense(64,activation='relu'))
#finally, our output layer should have only one neuron because we are trying to predict only one number
#notice how there is a different activation function in this layer
#this is because we want our outputs for each class to be a probability
model.add(Dense(nClasses,activation='softmax'))

#compile our model - set our loss metric (categorical cross entropy) and optimizer (stochastic gradient descent)
#how does the model performance change with different optimizers (ie AdaGrad, SGD, etc.)?
model.compile(loss='CategoricalCrossentropy',optimizer='Adam',metrics=['accuracy'])

#let's see a summary of our model
model.summary()


history = model.fit(
    df, labels,
    validation_split=0.3,
    verbose=1, epochs=100, batch_size=100, shuffle=True)


def show_loss(history):
    plt.figure()
    plt.plot(history.history['val_loss'], label="val loss")
    plt.plot(history.history['loss'],label="train loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
show_loss(history)


_, acc = model.evaluate(df_test, df_testLabels, verbose=1)
print('Accuracy for test data is',acc)

preds = model.predict(df_test)
preds = [i.round() for i in preds]
preds = tf.one_hot(tf.math.argmax(preds, axis = 1),depth = len(preds[0]))

preds = enc.inverse_transform(preds)
testLabels = enc.inverse_transform(df_testLabels)
cm = confusion_matrix(testLabels, preds)

_ =plt.imshow(cm, cmap=plt.cm.Blues)
_ = plt.xlabel("Predicted labels")
_ = plt.ylabel("True labels")

_ =plt.xticks(np.arange(0,len(np.unique(testLabels)),1),np.unique(testLabels))
_ =plt.yticks(np.arange(0,len(np.unique(testLabels)),1),np.unique(testLabels))
_ =plt.title('Confusion matrix ')
_ =plt.colorbar()
plt.show()

_=plt.hist(enc.inverse_transform(labels),label='training labels')
_=plt.hist(testLabels,label='test labels')
_=plt.yscale('log')
_=plt.legend()


