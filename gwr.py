import pandas as pd

print "Hello GWR"
import statsmodels
import os, sys, scipy
import numpy as np

print statsmodels.__version__
print scipy.__version__
print pd.__version__


# Import pygwr. Try first import of installed version.
# If it fails, we assume we try to run the script from inside the example folder.
# If this assumption is correct, we can simply add the parent folder to the
# Python path, and load it again.
try:
    import pygwr
except:
    sys.path.append(os.path.abspath('../'))
    import pygwr

# In Tokyomortality.txt file from
# http://www.st-andrews.ac.uk/geoinformatics/gwr/gwr-downloads/
# column are separated by several spaces. We need to fix this as it is not
# very handy for reading as simple CSV file.
# So we first convert this file into a standard Tab-separated file.

print "Starting..."

# Read now the data using the read_csv function in pygwr
h,data = pygwr.read_csv('C:/Users/tobiasrp/data/gwr_test.csv', header=True, sep="\t")

# Convert data into a Numpy array, make sure that the data are floats
data = np.array(data, dtype=np.float64)

# Separate data in dependent, independent, and location variables
y = data[:, h.index('log_price_plus_comdebt')]      # db2564 is the dependent variable
x = data[:, [h.index('build_year'), h.index('prom')]] # independent variables
g = data[:, [h.index('coord_x'), h.index('coord_y')]] # locations
ids = data[:, h.index('id')]    # list of IDs

# Create our GWR model
model = pygwr.GWR(targets=y, samples=x, locations=g)

# Make the global model first
print "Estimating global model..."
globalfit = model.global_regression()
print "Result for global model:"
print globalfit.summary()

# Make the bandwidth selection using simple interval search
# We use AICc as selection criterion
print "Estimating optimal bandwidth..."
bwmin, bwmax, bwstep = 5000, 20000, 1000
opt_bw, opt_aicc = None, np.inf     # initial values (AICc = infinity)
for bw in range(bwmin, bwmax+bwstep, bwstep):
    aicc = model.aicc(bw)   # calculate AICc (and AIC, BIC, deviance and K)
    print "   Bandwidth: %i -- AICc: %f" % (bw, aicc['aicc'])
    if aicc['aicc'] < opt_aicc: opt_bw, opt_aicc = bw, aicc['aicc']
print "   Optimal bandwidth is: %i" % opt_bw

# Estimate the GWR model at all data points
print "Estimating GWR model at all data points..."
gwr_result = model.estimate_at_target_locations(bandwidth=opt_bw)

# Write the result into a result file
gwr_result.export_csv('tokyomortality_gwr_result.csv')

print "Done."