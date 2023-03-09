# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 10:11:59 2022

@author: NTOU
"""

import re
import linecache
import numpy as np
import pandas as pd
import os
def text_read(filename):
    try:
        file = open(filename,'r')
    except IOError:
        error = []
        return error
    content = file.readlines()

    rows=len(content)
    datamat=np.zeros((rows,114))
    row_count=0
    
    for i in range(rows):
        content[i] = content[i].strip().split('\t')
        datamat[row_count,:] = content[i][:]
        row_count+=1

    file.close()
    return datamat
def text_read2(filename):
    try:
        file = open(filename,'r')
    except IOError:
        error = []
        return error
    content = file.readlines()

    rows=len(content)
    datamat=np.zeros((rows,5))
    row_count=0
    
    for i in range(rows):
        content[i] = content[i].strip().split('\t')
        datamat[row_count,:] = content[i][:]
        row_count+=1

    file.close()
    return datamat
x = text_read('fit_train_input_data.txt')
y = text_read('fit_train_output_data.txt')
xt = text_read2('fit_test_input_data.txt')
yt = text_read2('fit_test_output_data.txt')
train_data_x = x[0:5000,0:114]
train_data_y = y[0:3,0:114]
test_data_x = xt[0:5000,0:5]
test_data_y = yt[0:3,0:5]
size_tr = np.shape(train_data_x)
size_tr = np.array(size_tr)
size_tr1 = size_tr[0]
size_tr2 = size_tr[1]
index=np.arange(size_tr2)
np.random.shuffle(index)
train_data_x =train_data_x[:,index]
train_data_y =train_data_y[:,index]
train_data_x=np.transpose(train_data_x)
train_data_y=np.transpose(train_data_y)
test_data_x=np.transpose(test_data_x)
test_data_y=np.transpose(test_data_y)


#%%
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
from tensorflow.keras import backend as K  
model = Sequential()
model.add(Dense(100,input_dim=5000, activation = 'relu', name='dense_1'))
model.add(Dense(100, activation = 'relu', name='dense_2'))
model.add(Dense(100, activation = 'relu', name='dense_3'))
model.add(Dense(100, activation = 'relu', name='dense_4'))
model.add(Dense(100, activation = 'relu', name='dense_5'))
model.add(Dense(100, activation = 'relu', name='dense_6'))
model.add(Dense(100, activation = 'relu', name='dense_7'))
model.add(Dense(100, activation = 'relu', name='dense_8'))
model.add(Dense(3, name='dense_9'))
model.compile(optimizer = 'adam', loss = 'mae', metrics=['accuracy'])
history =model.fit(train_data_x, train_data_y,batch_size=48,validation_split=0.1, epochs = 100000,verbose=0)
model.save('d113_50m_va_mse_L8N100.h5')
#%%
import matplotlib.pyplot as plt
print(history.history.keys())
#%%
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
#%%
# summarize history for loss plt.plot(history.history['loss']) plt.plot(history.history['val_loss']) plt.title('model loss')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
#%%
accuracy=history.history['accuracy']
accuracy=np.array(accuracy)
val_accuracy=history.history['val_accuracy']
val_accuracy=np.array(val_accuracy)
loss=history.history['loss']
loss=np.array(loss)
val_loss=history.history['val_loss']
val_loss=np.array(val_loss)
#%%

model = Sequential()
model = keras.models.load_model('d113_50m_va_mse_L8N100.h5')
predict = model.predict(pd.DataFrame(test_data_x))


