import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tickster

# ticker = input('Enter ticker: \n')

all_tickers = pd.read_csv('tickers.csv')['Tickers']
for ticker in all_tickers:
    # for windows os: 
    all_data = pd.read_csv('C://Users//vasantgc//Documents//StockPredictions2.0//normalized_data//normalized_'+ticker+'.csv')
    # all_data = pd.read_csv('/Users/gcvasanta/Desktop/BetterStockPredictor/StockPredictions2.0/normalized_data/normalized_'+ticker+'.csv')
    
    date_data = pd.read_csv('C://Users//vasantgc//Documents//StockPredictions2.0//data//'+ticker+'.csv')['Date']
    # date_data = pd.read_csv('/Users/gcvasanta/Desktop/BetterStockPredictor/StockPredictions2.0/data/'+ticker+'.csv')['Date']

    open = list(all_data['Open'])
    close = list(all_data['Close'])
    high = list(all_data['High'])
    low = list(all_data['Low'])
    # open.reverse()
    # close.reverse()
    # high.reverse()
    # low.reverse()
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
    # input('stop')
    tick_spacing = 100
    y_axis = []
    for i in range(len(train_x)):
        y_axis.append(i)
    y_axis = np.array(y_axis)
    fig, ax = plt.subplots(1,1)
    ax.plot(date_data,close)
    ax.xaxis.set_major_locator(tickster.MultipleLocator(tick_spacing))
    plt.title(ticker)
    #for windows os: 
    plt.savefig('C://Users//vasantgc//Documents//StockPredictions2.0//graphs/'+ticker+'.png')
    # plt.savefig('/Users/gcvasanta/Desktop/BetterStockPredictor/StockPredictions2.0/graphs/'+ticker+'.png')
    plt.show()