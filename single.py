from matplotlib import pyplot as plt
from matplotlib import ticker as ticker
import numpy as np
import pandas as pd
data = (pd.read_csv('C://Users//vasantgc//Documents//StockPredictions2.0//data//hon.csv'))
dates = list(data['Date'])
open = list(data['Open'])
close = list(data['Close'])
dates.reverse()
open.reverse()
close.reverse()
fig, ax = plt.subplots(1,1)
ax.plot(dates,open)
ax.plot(dates,close)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
plt.xticks(rotation = 45)

plt.show()
print(data)
