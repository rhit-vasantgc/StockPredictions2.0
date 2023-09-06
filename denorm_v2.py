import pandas as pd
import numpy as np
# z times standard dev + mu
def denormalizer(rawX,meansArr,stdevArr):
    output = []
    for i in range(len(rawX)):
        # print(stdevArr[i+1])
        # print(means[i+1])
        output.append((rawX[i]*stdevArr[i+1])+meansArr[i+1])
    return output

ticks = list(pd.read_csv('tickers.csv')['Tickers'])
print(ticks)
# input(ticks.index('aapl'))
# ticks = ['aapl']
for i in range(len(ticks)):
    valLen = len(np.array(pd.read_csv('raw_future_predictions//'+ticks[i]+'.csv')))
    means = np.array((pd.read_csv('means.csv')).transpose()[i][1:5])
    stdevs = np.array(pd.read_csv('stdevs.csv').transpose()[i][1:5])
    # input(valLen)
    writeFile = open('denormalized_future_predictions//'+ticks[i]+'.csv','w')
    writeFile.write("Open,Close,High,Low\n")
    for j in range(valLen):
        # print(j)
        # print(ticks[i])
        values = np.array(pd.read_csv('raw_future_predictions//'+ticks[i]+'.csv').transpose()[j])
        # for value in values:
        # print(means)
        # print(stdevs)
        # print(values)
        # input('stop')
        acc = denormalizer(values,means,stdevs)
        # print(acc)
        writeFile.write(str(acc[0])+'\n')
    # writeFile.write((str(denormalizer(values,means,stdevs))))
# print(values)
