from scipy.spatial.distance import pdist, squareform
import numpy as np
import matplotlib.pyplot as plt

points = np.random.uniform(-.5, .5, (1000,2))

# Compute the distance between each different pair of points in X with pdist.
# Then, just for ease of working, convert to a typical symmetric distance matrix
# with squareform.
dists = squareform(pdist(points))

poi = points[4] # point of interest
dist_min = .4
close_points = dists[4] < dist_min

print close_points
print [i for i, x in enumerate(close_points) if x]

print("There are {} other points within a distance of {} from the point "
    "({:.3f}, {:.3f})".format(close_points.sum() - 1, dist_min, *poi))

f,ax = plt.subplots(subplot_kw=
    dict(aspect='equal', xlim=(-.5, .5), ylim=(-.5, .5)))
ax.plot(points[:,0], points[:,1], 'b+ ')
ax.plot(poi[0], poi[1], ms=15, marker='s', mfc='none', mec='g')
ax.plot(points[close_points,0], points[close_points,1],
    marker='o', mfc='none', mec='r', ls='')  # draw all points within distance

t = np.linspace(0, 2*np.pi, 512)
circle = dist_min*np.vstack([np.cos(t), np.sin(t)]).T
ax.plot((circle+poi)[:,0], (circle+poi)[:,1], 'k:') # Add a visual check for that distance
plt.show()