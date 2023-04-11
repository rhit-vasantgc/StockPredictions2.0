import pandas as pd
import numpy as np
ticker = input('Enter ticker: \n')
all_data = pd.read_csv('C://Users//vasantgc//Documents//StockPredictions2.0//normalized_data//normalized_'+ticker+'.csv')
open = list(all_data['Open'])
close = list(all_data['Close'])
high = list(all_data['High'])
low = list(all_data['Low'])
open.reverse()
close.reverse()
high.reverse()
low.reverse()
train_x = []
train_y = []
#5 days in advance
for i in range(len(open)-5):
    temp = []
    temp.append(open[i])
    temp.append(close[i])
    temp.append(high[i])
    temp.append(low[i])
    train_x.append(temp)
for i in range(5,len(open)):
    temp = []
    temp.append(open[i])
    temp.append(close[i])
    temp.append(high[i])
    temp.append(low[i])
    train_y.append(temp)
train_x = np.array(train_x)
train_y = np.array(train_y)


print(train_x)
print(train_x[4])
print(train_x[5])
print(train_x.shape)
print(train_y)
print(train_y[0])
print(train_y.shape)
print(all_data)