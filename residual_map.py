db_link = 'C:/Users/Tobias/data/csv_to_shp_test.csv'
shp_link = 'C:/Users/Tobias/data/shape_test.shp'

from pylab import *
import numpy as np
from pandas import DataFrame, Series, read_csv
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

z = read_csv(db_link)
z = np.array( z, dtype=np.float )
z = DataFrame( z, columns=['resids','coord_x','coord_y'] )

fig, ax = subplots()
ax.scatter(x = z.coord_x, y = z.coord_y, c=z.resids, cmap='RdYlGn', alpha = 0.8, linewidths = 0, marker = 'o' )
ax.set_aspect(0.5)
xlim(10.6135,10.959)
ylim(59.802,59.992)
xlabel('X')
ylabel('Y')
title('resids')

datafile = cbook.get_sample_data('C:/Users/Tobias/data/oslo.png')
img = plt.imread(datafile)
plt.imshow(img, zorder=0, extent=[10.6135, 10.959, 59.802, 59.992])

#[left, right, bottom, top]

plt.show()
