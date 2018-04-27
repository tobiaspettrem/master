db_link = 'C:/Users/tobiasrp/data/residual_adjusted_estimates.csv'
#shp_link = 'C:/Users/tobiasrp/data/shape_test.shp'

from pylab import *
import numpy as np
from pandas import DataFrame, Series, read_csv
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

z = read_csv(db_link)

z = z[["post_resid_adjustment_residual","coord_x","coord_y"]]
#post_resid_adjustment_residual
#pre_resid_adjustment_residual

print z.shape
z = z[np.abs(z.post_resid_adjustment_residual - z.post_resid_adjustment_residual.mean()) < (3*z.post_resid_adjustment_residual.std())]
print z.shape

z = np.array( z, dtype=np.float )
z = DataFrame( z, columns=['resids','coord_x','coord_y'] )

fig, ax = subplots()
ax.scatter(x = z.coord_x, y = z.coord_y, c=z.resids, cmap='RdYlGn', alpha = 0.8, linewidths = 0, marker = 'o' )
ax.set_aspect(1)
xlim(10.6135,10.959)
ylim(59.802,59.992)
xlabel('X')
ylabel('Y')
title('post_resid_adjustment_residual')

#datafile = cbook.get_sample_data('C:/Users/tobiasrp/data/oslo.png')
#img = plt.imread(datafile)
#plt.imshow(img, zorder=0, extent=[10.6135, 10.959, 59.802, 59.992])

#[left, right, bottom, top]

plt.show()
