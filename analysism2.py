import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
all_tickers = (pd.read_csv('tickers.csv')['Tickers'])

for ticker in all_tickers:
    print(ticker)
    data = np.array(pd.read_csv('C://Users//vasantgc//Documents//StockPredictions2.0//normalized_data//normalized_'+ticker+'.csv'))


    print(data)
    differences = [[],[],[],[]]
    for i in range(1,len(data)):
        for j in range(len(data[i])):
            differences[j].append(data[i][j]-data[i-1][j])
    print()
    print()
    a = np.transpose(np.array(differences))
    print(a)
    open = []


    for i in range(len(a)):
        open.append(a[i][0])


    xys = []
    for i in range(len(open)):
        xys.append(i)
    fig, ax = plt.subplots(1,1)
    ax.plot(xys,open)
    # ax.xaxis.set_major_locator(tickster.MultipleLocator(tick_spacing))
    print(open)
    plt.title(ticker)
    #for windows os: 
    plt.savefig('C://Users//vasantgc//Documents//StockPredictions2.0//difference_graphs/'+ticker+'.png')
    #plt.savefig('/Users/gcvasanta/Desktop/BetterStockPredictor/StockPredictions2.0/graphs/'+ticker+'.png')
    # plt.show()


#classifying differences methodology

# differences = [[],[],[],[]]
# for i in range(1,len(data)):
#     for j in range(len(data[i])):
#         # print(j)
#         # print(differences[j])
#         if(data[i][j]-data[i-1][j]>0):
#             differences[j].append(1)
#         else:
#             differences[j].append(-1)
# temp = np.array(differences)
# print(np.sum(temp[0]))
# print(np.sum(temp[1]))
# print(np.sum(temp[2]))
# print(np.sum(temp[3]))
# differences = np.transpose(np.array(differences))
# print(np.array(differences))
        
