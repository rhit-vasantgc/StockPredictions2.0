import sys
import json
import ast
import pandas as pd
import numpy as np
input = ast.literal_eval(sys.argv[1])
output = []
output = input
# print(input[0])
data_to_pass_back = np.ndarray.tolist(np.array(pd.read_csv('C://Users//vasantgc//Documents//StockPredictions2.0//denormalized_future_predictions//'+input[0]+'.csv')))
# data_to_pass_back = ['Sent from python']
output.append(data_to_pass_back)
print(json.dumps(output))
sys.stdout.flush()

