import pandas as pd

print "Hello GWR"
import gwr
import statsmodels
import os, sys, scipy
import numpy as np
import pysal as ps
import geopandas as gp
import matplotlib as plt

print statsmodels.__version__
print scipy.__version__
print pd.__version__

data = ps.open(ps.examples.get_path('GData_utm.csv'))
shp = gp.read_file('/Users/toshan/dev/pysal/pysal/examples/georgia/G_utm.shp')
vmin, vmax = np.min(shp['PctBach']), np.max(shp['PctBach'])
ax = shp.plot('PctBach', vmin=vmin, vmax=vmax, figsize=(8,8), cmap='Reds')
ax.set_title('PctBach' + ' T-vals')
fig = ax.get_figure()
cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap='Reds')
sm._A = []
fig.colorbar(sm, cax=cax)
