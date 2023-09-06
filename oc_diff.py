import pandas as pd
import numpy as np
import csv
def normalizer(data):
    mean = np.mean(data)
    stdev = np.std(data)
    output = []
    for i in data:
        output.append([(i-mean)/stdev])
    return output

ticks = np.array(pd.read_csv('tickers.csv')['Tickers'])
for tick in ticks:
    target_file = open('normalized_oc_diffs//'+tick+'.csv','w',newline='')
    target_file.write('normalized differences\n')
    opens = np.array(pd.read_csv('data//'+tick+'.csv')['Open'])
    closes = np.array(pd.read_csv('data//'+tick+'.csv')['Close'])
    raw_diffs = closes-opens
    normd = normalizer(raw_diffs)
    # print(normd)
    # input('stop')
    raw_diffs = np.ndarray.tolist(np.array(raw_diffs).reshape(len(raw_diffs),1))
    with target_file as file:
        writer = csv.writer(file)
        writer.writerows((normd))
# print(ticks)
# print()
# print(opens)
# print()
# print(closes)
# print()
# print(raw_diffs)