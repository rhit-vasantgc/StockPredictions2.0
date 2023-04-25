import pandas as pd
import numpy as np
data = np.array(pd.read_csv('C://Users//vasantgc//Documents//StockPredictions2.0//normalized_data//normalized_aapl.csv'))
positiveDiffs = [0,0,0,0]
negativeDiffs = [0,0,0,0]
maxStreakPositive = [0,0,0,0]
maxStreakNegative = [0,0,0,0]
for i in range(1,len(data)):
    for j in range(len(data[i])):
        # print(data[i][j])
        # print(data[i-1][j])
        if(data[i][j]>data[i-1][j]):
            positiveDiffs[j]+=1
            #print(positiveDiffs)
        else:
            negativeDiffs[j]+=1
        # input(j)

print(data)
print(positiveDiffs)
print(negativeDiffs)