
from pylab import *
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import random


def plot_districts(districts, district_column_name, data_set): #districts is a set of unique districts

    random.shuffle(districts)

    hot = plt.get_cmap('Paired')
    cNorm  = colors.Normalize(vmin=0, vmax=len(districts))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=hot)
    plt.ion()

    # Plot each species
    for i in range(len(districts)):
        index = pd.Series(data_set[[district_column_name]] == districts[i])
        plt.scatter(data_set.coord_x[index], data_set.coord_y[index], s=10, color=scalarMap.to_rgba(i), label=districts[i])

    plt.draw()
    plt.pause(0.2)
    print "Updating plot"


"""
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

