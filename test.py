db_link = 'C:/Users/tobiasrp/data/training.csv'

from pylab import *
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import random

virdi = pd.read_csv(db_link)
virdi = virdi[["kr/m2", "coord_x", "coord_y", "bydel_code", "kmeans_cluster"]]

X_MAX = virdi.coord_x.max()
X_MIN = virdi.coord_x.min()
Y_MAX = virdi.coord_y.max()
Y_MIN = virdi.coord_y.min()
SQMPRICE_MAX = virdi['kr/m2'].max()
SQMPRICE_MIN = virdi['kr/m2'].min()

def init_kmeans():
    x = random.uniform(X_MIN, X_MAX)
    y = random.uniform(Y_MIN, Y_MAX)
    sqmprice = random.uniform(SQMPRICE_MIN, SQMPRICE_MAX)

    return (x, y, sqmprice)

def kmeans(df, k):
    centroids = [init_kmeans() for i in range(k)]


kmeans(virdi,20)

print list(set(virdi[["kmeans_cluster"]].iloc[:,0]))

districts = list(set(virdi[["kmeans_cluster"]].iloc[:,0]))

random.shuffle(districts)

hot = plt.get_cmap('Paired')
cNorm  = colors.Normalize(vmin=0, vmax=len(districts))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=hot)
plt.ion()


for i in range(len(districts)):
    indx = virdi[["kmeans_cluster"]].iloc[:,0] == districts[i]
    plt.scatter(virdi.coord_x[indx], virdi.coord_y[indx], s=10, color=scalarMap.to_rgba(i), label=districts[i])

plt.draw()
plt.pause(0.2)
print "Updating plot"
"""
random.shuffle(districts)

for i in range(len(districts)):
    indx = virdi.bydel_code == districts[i]
    plt.scatter(virdi.coord_x[indx], virdi.coord_y[indx], s=50, color=scalarMap.to_rgba(i), label=districts[i])
plt.draw()
plt.pause(0.2)
print "Updating plot"

random.shuffle(districts)

for i in range(len(districts)):
    indx = virdi.bydel_code == districts[i]
    plt.scatter(virdi.coord_x[indx], virdi.coord_y[indx], s=50, color=scalarMap.to_rgba(i), label=districts[i])
plt.draw()
plt.pause(0.2)
print "Updating plot"

random.shuffle(districts)

for i in range(len(districts)):
    indx = virdi.bydel_code == districts[i]
    plt.scatter(virdi.coord_x[indx], virdi.coord_y[indx], s=50, color=scalarMap.to_rgba(i), label=districts[i])
plt.draw()
plt.pause(0.2)
print "Updating plot"

random.shuffle(districts)

for i in range(len(districts)):
    indx = virdi.bydel_code == districts[i]
    plt.scatter(virdi.coord_x[indx], virdi.coord_y[indx], s=50, color=scalarMap.to_rgba(i), label=districts[i])

fig, ax = subplot()
ax.scatter(x = virdi.coord_x, y = virdi.coord_y, c=virdi.bydel_code, cmap='RdYlGn', alpha = 0.8, linewidths = 0, marker = 'o' )
ax.set_aspect(1)
xlim(10.6135,10.959)
ylim(59.802,59.992)
xlabel('X')
ylabel('Y')
title('Bydeler')
"""
#datafile = cbook.get_sample_data('C:/Users/tobiasrp/data/oslo.png')
#img = plt.imread(datafile)
#plt.imshow(img, zorder=0, extent=[10.6135, 10.959, 59.802, 59.992])

#[left, right, bottom, top]
