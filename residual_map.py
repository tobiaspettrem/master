db_link = 'C:/Users/tobiasrp/data/residual_adjusted_estimates.csv'
#shp_link = 'C:/Users/tobiasrp/data/shape_test.shp'

from pylab import *
import numpy as np
from pandas import DataFrame, Series, read_csv
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

z = read_csv(db_link)
print z.head()
z = z[["regression_residual","coord_x","coord_y"]]

z = z.sample(frac = 0.9)

#post_resid_adjustment_residual
#pre_resid_adjustment_residual

print z.shape
z = z[np.abs(z.regression_residual - z.regression_residual.mean()) < (3*z.regression_residual.std())]
print z.shape

z = np.array( z, dtype=np.float )
z = DataFrame( z, columns=['regression_residual','coord_x','coord_y'] )

fig, ax = subplots()
ax.scatter(x = z.coord_x, y = z.coord_y, c=z.regression_residual, cmap='RdYlGn', s = 20, alpha = 0.8, linewidths = 0, marker = 'o' )
ax.set_aspect(1)
xlim(10.6135,10.959)
ylim(59.802,59.992)
xlabel('X')
ylabel('Y')
title('regression_residual')

#datafile = cbook.get_sample_data('C:/Users/tobiasrp/data/oslo.png')
#img = plt.imread(datafile)
#plt.imshow(img, zorder=0, extent=[10.6135, 10.959, 59.802, 59.992])

#[left, right, bottom, top]

plt.show()
