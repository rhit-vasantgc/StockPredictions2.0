import pandas as pd
import numpy as np
from tensorflow import keras
import math
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout
from tensorflow.keras.backend import sigmoid
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation
import tensorflow as tf
import subprocess



def swish(x, beta = 1):
    # 0.1*tf.math.exp(0.1*x)*tf.math.sin(10*x) <-- original function that works great (mostly)
    return 0.1375*tf.math.exp(0.1*x)*tf.math.sin(10*x)
get_custom_objects().update({'swish': Activation(swish)})
ticks = np.array(pd.read_csv('tickers.csv')['Tickers'])
numDaysForward = int(input('Enter the number of days into the future you want to predict:\n'))

# tick = 'aapl'
# normalized_diff_data = pd.read_csv("normalized_data//normalized_"+tick+".csv")
#     #writeFile = open('')
# train_data = [normalized_diff_data["Open"][0:len(normalized_diff_data)-numDaysForward],
# normalized_diff_data["Close"][0:len(normalized_diff_data)-numDaysForward],
# normalized_diff_data["High"][0:len(normalized_diff_data)-numDaysForward],
# normalized_diff_data["Low"][0:len(normalized_diff_data)-numDaysForward]]
# test_data = [normalized_diff_data["Open"][numDaysForward:len(normalized_diff_data)],
# normalized_diff_data["Close"][numDaysForward:len(normalized_diff_data)],
# normalized_diff_data["High"][numDaysForward:len(normalized_diff_data)],
# normalized_diff_data["Low"][numDaysForward:len(normalized_diff_data)]]

# train_data = np.transpose(np.array(train_data))
# test_data = np.transpose(np.array(test_data))

# train_data = train_data.reshape(np.shape(train_data)[0],1,4)
# test_data = test_data.reshape(np.shape(test_data)[0],1,4)
# model = Sequential()
# model.add((LSTM(len(train_data),activation='swish',return_sequences=True,input_shape=(1,4))))
# model.add(Dropout(0.1))
# model.add((LSTM(710,activation='sigmoid',return_sequences=True)))
# model.add(Dense(4))
# #lr is 0.001 tanh -> sigmoid 5 epochs
# learning_rate = 0.0005
# optimizer = tf.keras.optimizers.Adam(0.001)
# optimizer.learning_rate.assign(learning_rate)

# model.compile(optimizer=optimizer, loss='mae', metrics=['mse','acc'])
# history = model.fit(train_data, test_data, epochs=15, verbose=1)

# modelFile = open('models//'+tick+'.csv','w')
# modelFile.write('Model Object Reference\n')
# modelFile.write(str(model))
# writeFile = open('predicted_data//'+tick+'.csv','w')
# writeFile.write('Predicted Open,Predicted Close,Predicted High,Predicted Low,Actual Open,Actual Close,Actual High,Actual Low\n')
# predictedFile = open('raw_future_predictions//'+tick+'.csv','w')
# predictedFile.write('Predicted Open,Predicted Close,Predicted High,Predicted Low\n')
# rands = random.sample(range(0,len(train_data)),5)
# for b in rands:

#     print('Tick: ' + tick + ' index value ' + str(b))
#     pred = model.predict(train_data[b].reshape(1,1,4))[0][0]
#         # print(pred)
#     for a in range(0,4):
#         writeFile.write(str(model.predict(train_data[b].reshape(1,1,4))[0][0][a])+',')
#     for a in range(0,3):
#         writeFile.write(str(test_data[b][0][a])+',')
#     writeFile.write(str(test_data[b][0][3])+'\n')
# temp = test_data[len(test_data)-numDaysForward:len(test_data)]
# for i in temp:
#     predict = model.predict(i.reshape(1,1,4))[0][0]
#     for g in range(0,3):
#         predictedFile.write(str(predict[g])+',')
#     predictedFile.write(str(predict[3])+'\n')
#     print(predict)
# input('end')
# modelObjects = {}
for tick in ticks:
    if(tick == 'aapl'):
        print('need to skip')
        continue
    normalized_diff_data = pd.read_csv("normalized_data//normalized_"+tick+".csv")
    #writeFile = open('')
    train_data = [normalized_diff_data["Open"][0:len(normalized_diff_data)-numDaysForward],
    normalized_diff_data["Close"][0:len(normalized_diff_data)-numDaysForward],
    normalized_diff_data["High"][0:len(normalized_diff_data)-numDaysForward],
    normalized_diff_data["Low"][0:len(normalized_diff_data)-numDaysForward]]

    test_data = [normalized_diff_data["Open"][numDaysForward:len(normalized_diff_data)],
    normalized_diff_data["Close"][numDaysForward:len(normalized_diff_data)],
    normalized_diff_data["High"][numDaysForward:len(normalized_diff_data)],
    normalized_diff_data["Low"][numDaysForward:len(normalized_diff_data)]]

    train_data = np.transpose(np.array(train_data))
    test_data = np.transpose(np.array(test_data))

    train_data = train_data.reshape(np.shape(train_data)[0],1,4)
    test_data = test_data.reshape(np.shape(test_data)[0],1,4)



    model = Sequential()
    # model.add((LSTM(len(train_data),activation='swish',return_sequences=True,input_shape=(1,4))))
    # model.add((LSTM(750,activation='sigmoid',return_sequences=True)))
    #learning_rate = 0.0005 15 or 10 epochs
    model.add((LSTM(len(train_data),activation='swish',return_sequences=True,input_shape=(1,4))))
    # model.add(Dropout(0.1))
    model.add((LSTM(750,activation='swish',return_sequences=True)))
    model.add(Dropout(0.1))
    model.add((LSTM(300,activation='sigmoid',return_sequences=True)))
    
    model.add(Dense(4))

    #lr is 0.001 tanh -> sigmoid 5 epochs
    learning_rate = 0.0005
    optimizer = tf.keras.optimizers.Adam(0.001)
    optimizer.learning_rate.assign(learning_rate)

    model.compile(optimizer=optimizer, loss='mae', metrics=['mse','acc'])
    history = model.fit(train_data, test_data, epochs=10, verbose=1)
    losses = history.history['loss']
    mses = history.history['mse']
    acc = history.history['acc']
    metricsArr = np.array([losses,mses,acc])
    metricsArr = metricsArr.transpose()

    print(metricsArr)
    metricsFile = open('metrics//'+tick+'.csv','w')
    metricsFile.write('MAE Loss,MSE Loss,Accuracy\n')
    for s in range(len(metricsArr)):
        aux = metricsArr[s]
        for g in range(len(aux)-1):
            metricsFile.write(str(aux[g])+',')
        metricsFile.write(str(aux[len(aux)-1])+'\n')
    # input('stop')
    modelFile = open('models//'+tick+'.csv','w')
    modelFile.write('Model Object Reference\n')
    modelFile.write(str(model))
    writeFile = open('predicted_data//'+tick+'.csv','w')
    writeFile.write('Predicted Open,Predicted Close,Predicted High,Predicted Low,Actual Open,Actual Close,Actual High,Actual Low\n')
    predictedFile = open('raw_future_predictions//'+tick+'.csv','w')
    predictedFile.write('Predicted Open,Predicted Close,Predicted High,Predicted Low\n')
    rands = random.sample(range(0,len(train_data)),5)
    for b in rands:

        print('Tick: ' + tick + ' index value ' + str(b))
        pred = model.predict(train_data[b].reshape(1,1,4))[0][0]
        # print(pred)
        for a in range(0,4):
            writeFile.write(str(model.predict(train_data[b].reshape(1,1,4))[0][0][a])+',')
        for a in range(0,3):
            writeFile.write(str(test_data[b][0][a])+',')
        writeFile.write(str(test_data[b][0][3])+'\n')
    temp = test_data[len(test_data)-numDaysForward:len(test_data)]
    for i in temp:
        predict = model.predict(i.reshape(1,1,4))[0][0]
        for g in range(0,3):
            predictedFile.write(str(predict[g])+',')
        predictedFile.write(str(predict[3])+'\n')
        print(predict)
    # input('end')
    # input('stop')
    
    # input(tick)
        # print(model.predict(train_data[b].reshape(1,1,4))[0][0])
        # print()
        # print(test_data[b][0])
        # # print(test_data[1])
        # print()
        # print(np.array(train_data)[b][0])
        # print(np.array(test_data)[b][0])
        # print()
        # print()
        # print()
        # 
# subprocess.run(["python","denormalizer.py"])