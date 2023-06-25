import pandas as pd
import numpy as np
def normalizer(means_array,std_array,x_array,ticker):
    #means_array and std_array are the means array and stdev array
    #x_array is the full 5 by _____ list in the form [[Opens],[Closes],[Highs],[Lows]]
    #we are not normalizing dates so in the loops below, we only use the first 4 values of this list  (Opens, Closes, Highs, Lows)
    #ticker is the ticker that this method needs to normalize

    #for windows os: 
    # normalizedFile = open('C://Users//vasantgc//Documents//StockPredictions2.0//normalized_data//normalized_'+ticker+'.csv','w')
    normalizedFile = open('/Users/gcvasanta/Desktop/BetterStockPredictor/StockPredictions2.0/normalized_data/normalized_'+ticker+'.csv','w')
    #open the normalized csv file
    normalizedFile.write('Open,Close,High,Low\n')
    #write headers
    arrayToWrite = []
    for i in range(len(x_array)-1):
        #i iterates through Open, Close, High, Low
        normalized_x_array = []
        for x in x_array[i]:
            #x iterates through the numbers themselves
            temp = x-means_array[i]
            normalized_x_array.append((temp/std_array[i]))
            #^ normalizes and appends auxillary array that temporary stores all normalized values for that column 
        arrayToWrite.append(normalized_x_array)
        #^append the column auxillary array to the full auxillary array
    arrayToWrite = np.transpose(np.array(arrayToWrite))
    #^ transposes the full auxillary array to make writing into csv file easier
    for norms in arrayToWrite:
        #^iterating through every row to add
        for s in range(len(norms)-1):
            normalizedFile.write(str(norms[s])+',')
            #^ iterating through the elements and writing them; did everything but last one because last one doesnt need a comma
        normalizedFile.write(str(norms[s+1])+'\n')
        #^ add newline after last zscore
tickers = np.array(pd.read_csv('tickers.csv')['Tickers'])
#means and stdevs are dictionaries that map the ticker to a list with [open mean,close mean,high mean,low mean] 
means = {}
stdevs = {}
for tick in tickers:
    #iterating through tickers to populate means and stdevs dictionaries

    #for windows os: 
    # data = np.transpose(np.array(pd.read_csv('C://Users//vasantgc//Documents//StockPredictions2.0//data//'+tick+".csv")))
    data = np.transpose(np.array(pd.read_csv('/Users/gcvasanta/Desktop/BetterStockPredictor/StockPredictions2.0/data/'+tick+".csv")))
    #getting data and transposing to make each element a column; this allows us to feed into normalizer method
    aux_mean = []
    aux_std = []
    #mean list and stdev list that a ticker points to 
    for b in range(0,4):
        aux_mean.append(np.mean(data[b]))
        aux_std.append(np.std(data[b]))
        #populating mean list and stdev list
    means[tick] = aux_mean
    stdevs[tick] = aux_std
    #assigning ticker to mean and stdev list
print(means)
print(stdevs)
meansFile = open('means.csv','w')
stdevsFile = open('stdevs.csv','w')
meansFile.write('ticker,open,close,high,low\n')
stdevsFile.write('ticker,open,close,high,low\n')
for tick in tickers:
    #for windows os: 
    # data = np.transpose(np.array(pd.read_csv('C://Users//vasantgc//Documents//StockPredictions2.0//data//'+tick+".csv")))
    data = np.transpose(np.array(pd.read_csv('/Users/gcvasanta/Desktop/BetterStockPredictor/StockPredictions2.0/data/'+tick+".csv")))
    normalizer(means[tick],stdevs[tick],data,tick)
    meansFile.write(tick+','+str(means[tick][0])+','+str(means[tick][1])+','+str(means[tick][2])+','+str(means[tick][3])+'\n')
    stdevsFile.write(tick+','+str(stdevs[tick][0])+','+str(stdevs[tick][1])+','+str(stdevs[tick][2])+','+str(stdevs[tick][3])+'\n')
    #using normalizer method to create normalized csv files

    
