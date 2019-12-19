from sklearn.cluster import KMeans
from sklearn import metrics
from sys import maxsize
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pylab as plt
import numpy as np


fig = plt.scatter([1, 2],[1, 2],
                  c=(0.2, 0.45), s=[5, 20])

plt.axis('off')
plt.show()
