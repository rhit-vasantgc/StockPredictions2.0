import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
previous = pd.read_csv('data//aapl.csv')
predictions = pd.read_csv('denormalized_future_predictions//aapl.csv')
print(previous)
print(predictions)
timesteps = []
future_timesteps = []
i = 0
for i in range(2000,len(previous['Close'])):
    timesteps.append(i)
for j in range(len(predictions['Close'])):
    future_timesteps.append(i+j)
plt.plot(timesteps,previous['Close'][2000::],color='red',label='Known Values')

plt.plot(future_timesteps,predictions['Close'],color='green',label='Future Prediction Values')
leg1 = plt.legend()
plt.savefig('apple_plot1.png')
plt.show()