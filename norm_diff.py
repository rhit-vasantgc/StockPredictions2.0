import pandas as pd
import numpy as np
all_tickers = pd.read_csv('tickers.csv')['Tickers']
print(all_tickers)
for ticker in all_tickers:
    #for windows os:
    data = np.array(pd.read_csv('C://Users//vasantgc//Documents//StockPredictions2.0//normalized_data//normalized_'+ticker+'.csv'))
    # data = np.array(pd.read_csv('/Users/gcvasanta/Desktop/BetterStockPredictor/StockPredictions2.0/normalized_data/normalized_'+ticker+'.csv'))
    #for windows os:
    diff_file = open('C://Users//vasantgc//Documents//StockPredictions2.0//normalized_difference_data//'+ticker+'.csv','w')
    # diff_file = open('/Users/gcvasanta/Desktop/BetterStockPredictor/StockPredictions2.0/normalized_difference_data/'+ticker+'.csv','w')
    diff_file.write('Open,Close,High,Low\n')
    for i in range(1,len(data)):
        diff_file.write(str(data[i][0]-data[i-1][0])+','+str(data[i][1]-data[i-1][1])+','+str(data[i][2]-data[i-1][2])+','+str(data[i][3]-data[i-1][3])+'\n')
       
            
        