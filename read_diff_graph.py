import matplotlib.pyplot as plt
import matplotlib.image as mpimg
ticker = input('enter ticker name; ')
path = 'C://Users//vasantgc//Documents//StockPredictions2.0//difference_graphs/'+ticker+'.png'
image = mpimg.imread(path)
plt.imshow(image)
plt.show()
#possible activation function: (copied from desmos) y=.02e^{.1x}\sin\left(10x\right)\ 