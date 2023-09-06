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
from tensorflow.keras.optimizers import Adamax
import tensorflow as tf
import subprocess
import sys

#.1375
tf.compat.v1.enable_eager_execution()
def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr
def rmsd(y_true,y_pred):
    # print(y_true[0][0][0])
    # print(y_pred[0][0][0])
    # print(y_true[1][0][0])
    sum = tf.constant(0.0)
    for i in range(0,103):
        # said index out of bounds for index 103 for some reason for a batch size of 128
        # print(y_true[i][0][0])
        sum = sum + tf.square(y_true[i][0][0]-y_pred[i][0][0])
    # print(sum)
    # div = tf.divide(sum,tf.constant(103.0))
    # print(div)
    return tf.sqrt(sum)
def rmse(y_true,y_pred): 
    # squared_difference = tf.square(y_true - y_pred)*tf.square(y_true-y_pred)
    #when the delta was 1, everything worked pretty well
    delta = 0.5
    delta_tensor = tf.constant([float(delta),float(delta),float(delta),float(delta)])
    # print(tf.math.greater(tf.math.abs(y_true-y_pred),delta_tensor))
    # input('stop')
    
    # print(tf.math.abs(y_true-y_pred).numpy())
    # input('stop above')
    # print(tf.where(tf.math.greater(tf.math.abs(y_true-y_pred),delta_tensor)).numpy())
    # print(np.any(tf.where(tf.math.greater(delta_tensor,tf.math.abs(y_true-y_pred))).numpy()))
    # input('stop')
    error = tf.subtract(y_true, y_pred)
    quantile = 0.5
    return tf.reduce_mean(tf.maximum(quantile*error, (quantile-1)*error), axis=-1)

    # if(np.any(tf.where(tf.math.greater(delta_tensor,tf.math.abs(y_true-y_pred))).numpy())):
    #     squared_difference = 0.5*tf.square(y_true-y_pred)
    # else:
    #     squared_difference = delta*(tf.math.abs(y_true-y_pred)-0.5*delta) 
    # return tf.reduce_mean(squared_difference, axis=1)


    # squared_difference = tf.math.square((tf.math.square(y_true) - tf.math.square(y_pred)))
    # print(squared_difference)
    # squared_difference = tf.sqrt(tf.math.abs(y_true-y_pred), name=None)
    

# tf.config.run_functions_eagerly(True)
# @tf.function
def swish(x, beta = 1):
    # 0.1*tf.math.exp(0.1*x)*tf.math.sin(10*x) <-- original function that works great (mostly)
    # 0.175\cdot e^{0.01x}\cdot\sin\left(10x\right) <-- been using this more as of 6/3/2023
    #0.175, 0.1, 10
    #0.1 before as of 6/8/2023
    #.125
   
    # print(tf.executing_eagerly())
    # print((0.125*tf.math.exp(0.01*x)*tf.math.sin(10*x)))
    # input('stop')
    return (0.020*tf.math.exp(0.0025*x)*(tf.math.sin(15*x)+0.175*x))
def swishhigher(x, beta = 1):
    # return (0.0825*tf.math.exp(0.1*x)*tf.math.sin(10*x))
    return (0.0275*tf.math.exp(0.0025*x)*(tf.math.sin(37.5*x)+0.175*x)) #.0025, .3, either 125 or 89.5
def swishlower(x, beta = 1):
    return (0.0125*tf.math.exp(0.0025*x)*(tf.math.sin(30*x)+0.175*x)) #37.5
get_custom_objects().update({'swish': Activation(swish)})
get_custom_objects().update({'swishhigher': Activation(swishhigher)})
get_custom_objects().update({'swishlower': Activation(swishlower)})
ticks = np.array(pd.read_csv('tickers.csv')['Tickers'])
# numDaysForward = int(input('Enter the number of days into the future you want to predict:\n'))
numDaysForward = 10
epochs = 75
# ticks = ['mmm']
for tick in ticks:
    # if(tick == 'aapl'):
    #     print('need to skip')
    #     continues
    normalized_diff_data = pd.read_csv("normalized_data//normalized_"+tick+".csv")
    ndf = pd.read_csv('normalized_difference_data//'+tick+'.csv')
    ocdiff = pd.read_csv('normalized_oc_diffs//'+tick+'.csv')
    # print(len(ndf))
    # print(len(normalized_diff_data))
    # input('stop')
    #writeFile = open('')
    train_x_data = [normalized_diff_data["Open"][0:len(normalized_diff_data)-numDaysForward-1],
    normalized_diff_data["Close"][0:len(normalized_diff_data)-numDaysForward-1],
    normalized_diff_data["High"][0:len(normalized_diff_data)-numDaysForward-1],
    normalized_diff_data["Low"][0:len(normalized_diff_data)-numDaysForward-1],
    ndf["Open"][0:len(ndf)-numDaysForward],
    ndf["Close"][0:len(ndf)-numDaysForward],
    ndf["High"][0:len(ndf)-numDaysForward],
    ndf["Low"][0:len(ndf)-numDaysForward],
    ocdiff['normalized differences'][0:len(ocdiff)-numDaysForward-1]]

    future_predict_data = [normalized_diff_data["Open"][len(normalized_diff_data)-numDaysForward:len(normalized_diff_data)],
    normalized_diff_data["Close"][len(normalized_diff_data)-numDaysForward:len(normalized_diff_data)],
    normalized_diff_data["High"][len(normalized_diff_data)-numDaysForward:len(normalized_diff_data)],
    normalized_diff_data["Low"][len(normalized_diff_data)-numDaysForward:len(normalized_diff_data)],
    ndf["Open"][len(ndf)-numDaysForward:len(ndf)],
    ndf["Close"][len(ndf)-numDaysForward:len(ndf)],
    ndf["High"][len(ndf)-numDaysForward:len(ndf)],
    ndf["Low"][len(ndf)-numDaysForward:len(ndf)],
    ocdiff['normalized differences'][len(ocdiff) - numDaysForward:len(ocdiff)]]
    
    train_y_data = [
    normalized_diff_data["Close"][numDaysForward:len(normalized_diff_data)-1]] 

    train_x_data = np.transpose(np.array(train_x_data))
    train_y_data = np.transpose(np.array(train_y_data))
    future_predict_data = np.transpose(np.array(future_predict_data))

    # print(train_data)
    # print(np.shape(train_data))
    # input('stop')
    train_x_data = train_x_data.reshape(np.shape(train_x_data)[0],1,9)
    train_y_data = train_y_data.reshape(np.shape(train_y_data)[0],1,1)
    
    higher = ['amgn','aapl','ba','csco','hd','intc','jnj','mcd','mrk','msft','nke','pg','trv','v']
    lower = ['crm','cvx','dis','hon','ko','mmm','unh','vz','wba','wmt']
    normal = ['axp','cat','dow','gs','jpm']
    activation = 'swish'
    if(tick in higher):
        activation='swishhigher'
    elif(tick in lower):
        activation='swishlower'
    model = Sequential()
    # model.add((LSTM(len(train_data),activation='swish',return_sequences=True,input_shape=(1,4))))
    # model.add((LSTM(750,activation='sigmoid',return_sequences=True)))
    #learning_rate = 0.0005 15 or 10 epochs
    model.add((Dense(int(len(train_x_data)/5)*0.9,activation=activation,input_shape=(1,9))))
    model.add((Dense(250*5,activation=activation)))
    # model.add(Dropout(0.1))
    # model.add(Dropout(0.25))
    model.add((Dense(75*5,activation='sigmoid')))
    #1000 to 250
    #2000 to 500
    # model.add((LSTM(225,activation='tanh',return_sequences=True)))
    model.add(Dense(1))

    #lr is 0.001 tanh -> sigmoid 5 epochs
    #0.0005
    # learning_rate = 0.0005
    # optimizer = tf.keras.optimizers.Adam(0.001)
    # optimizer.learning_rate.assign(learning_rate)

    # optimizer = 'Adamax'
    optimizer = tf.keras.optimizers.Adamax()

    # optimizer = tf.keras.optimizers.Adamax()
    # 0.001
    # optimizer = tf.keras.optimizers.experimental.SGD(0.01,momentum = 0.99)
    # optimizer = tf.keras.optimizers.experimental.RMSprop(learning_rate=0.001,momentum = 0.50)
    # lr_metric = get_lr_metric(optimizer)
    # ,lr_metric
    #loss is mae
    # tf.keras.losses.MeanAbsolutePercentageError()
    model.compile(optimizer=optimizer, loss=rmsd, metrics=['mse','acc',tf.keras.metrics.RootMeanSquaredError()],run_eagerly=True)
    print(model.optimizer.learning_rate.numpy())
    print(tf.keras.optimizers.schedules.LearningRateSchedule)
    # input('stop')
    # 25 epochs
    # amgn: learning rate is too high
    history = model.fit(train_x_data, train_y_data, epochs=epochs, verbose=1, batch_size=128)
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
    predictedFile.write('Predicted Open\n')
    rands = random.sample(range(0,len(train_x_data)),15)
    # rands = [1,550,1100,1630,1640,1641,1642,1643,1644,1645,1646,1647,1648,1649,1650,1651,1655,1660,1670,2200,2201,2202,2203,2204,2205]
    for b in rands:

        print('Tick: ' + tick + ' index value ' + str(b))
        pred = model.predict(train_x_data[b].reshape(1,1,9))[0][0]

        print(pred)
        print(train_y_data[b])
    # temp = train_y_data[len(train_y_data)-numDaysForward-1:len(train_y_data)-1]
    # print(np.array(future_predict_data).shape)
    # print()
    # print(temp)
    future_predict_data = np.array(future_predict_data)
    for fut in future_predict_data:
        print(fut)
        print()
        print(model.predict(fut.reshape(1,1,9))[0][0])
        predictedFile.write(str(model.predict(fut.reshape(1,1,9))[0][0][0])+'\n')
    #     # print(pred)
    #     for a in range(0,1):
    #         writeFile.write(str(model.predict(train_data[b].reshape(1,1,4))[0][0][a])+',')
    #     for a in range(0,3):
    #         writeFile.write(str(test_data[b][0][a])+',')
    #     writeFile.write(str(test_data[b][0][3])+'\n')
    # temp = test_data[len(test_data)-numDaysForward:len(test_data)]
    # # diffs_file = open('pred_diffs//'+tick+'.csv','w')
    # # diffs_file.write('open,close,high,low\n')
    # for i in temp:
    #     predict = model.predict(i.reshape(1,1,4))[0][0]
      
    #     for g in range(0,3):
    #         predictedFile.write(str(predict[g])+',')
    #     predictedFile.write(str(predict[3])+'\n')
    #     print(predict)
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