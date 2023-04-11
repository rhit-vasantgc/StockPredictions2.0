# def static_normalizer(x,mean,stdev):
#     return (x-mean)/stdev

# print(static_normalizer(101.31,173.21554022988528,28.541484299498123))
# print(static_normalizer(104.23,173.15627494252882,28.545915967884117))
# print(static_normalizer(104.43,174.52032331034482,28.565358684853262))
# print(static_normalizer(101.51,171.72748124137962,28.488126089155806))

# # 104.43

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

x = [0,5,9,10,15]
y = [0,1,2,3,4]

tick_spacing = 2

fig, ax = plt.subplots(1,1)
ax.plot(x,y)
ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
plt.show()