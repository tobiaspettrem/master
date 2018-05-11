import time
import matplotlib.pyplot as plt
import random
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

K = 16
PRICE_FACTOR = 1.0/20
PLOT_MODULO_VALUE = 1000
plt.ion()

"""
plot_X = random.sample(X, int(len(X) / 20))

N = 5000
X = X[:N]
print "Length: " + str(len(X))
"""
def cluster_points(X, mu):
    clusters  = {}

    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x[0:3]-mu[i[0]][0:3])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters

def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu

def has_converged(mu, oldmu):
    return set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu])


def update_plot(mu, clusters):
    plt.clf()

    cluster_x = [mu[i][0] for i in range(len(mu))]
    cluster_y = [mu[i][1] for i in range(len(mu))]

    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'tan', 'gold', 'cyan', 'violet', 'salmon', 'coral', 'sienna', 'skyblue','teal', 'pink']

    for i in range(K):
        points = np.array([plot_X[j] for j in range(len(plot_X)) if plot_X[j].tolist() in np.array(clusters[i]).tolist()])
        plt.scatter(points[:,0], points[:,1], s=30, linewidths=0, c=colors[i])
    plt.scatter(x = cluster_x, y = cluster_y, marker='*', s=200, c='#050505')

    plt.draw()
    plt.pause(0.1)

def find_centers(X, K):
    # Initialize to K random centers
    all_mu = []
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)
    count = 0
    while not has_converged(mu, oldmu):
        start = time.time()
        count += 1
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)

        if count % PLOT_MODULO_VALUE == 0:
            update_plot(mu, clusters)

        print "Iteration: " + str(count) + ". Time elapsed: " + str(round(time.time() - start, 2)) + " seconds."

        """
        if np.array(mu).tolist() in all_mu: #algorithm is repeating itself
            oldmu = random.sample(X, K)
            mu = random.sample(X, K)
            print "Repeating itself"

        all_mu.append(np.array(mu).tolist())
        """
    return(mu, clusters)

def add_kmeans_districts(df, K):

    print "K-means with K: " + str(K)

    df = df.assign(adjusted_sqm_price = np.log(df["kr/m2"]) * PRICE_FACTOR)
    X = np.array(df[["coord_x", "coord_y","adjusted_sqm_price","id"]])
    mu, clusters = find_centers(X, K)
    clustering_list = []
    for key in clusters:
        for points in clusters[key]:
            row = [key]
            row += [p for p in points]
            clustering_list.append(row)

    clustering = pd.DataFrame(clustering_list)
    clustering = clustering.rename(columns = {4:"id"})
    clustering = clustering.rename(columns = {0:"kmeans_cluster"})
    clustering = clustering[["id","kmeans_cluster"]]

    df = pd.merge(df, pd.DataFrame(clustering), on = "id")
    return df


def predict_kmeans_districts(test, training, KMEANS_K):

    X = training[["coord_x", "coord_y"]]
    y = training[["kmeans_cluster"]]

    neigh = KNeighborsClassifier(n_neighbors=KMEANS_K)
    neigh.fit(X, y)

    test = test.assign(kmeans_cluster_prediction = neigh.predict(test[["coord_x", "coord_y"]]))
    return test

"""
mu, clusters = find_centers(X, K)
print "Algorithm terminated"
update_plot(mu, clusters)
plt.pause(0.1)

l = []
for key in clusters:
    for points in clusters[key]:
        sub_list = [key]
        sub_list += [p for p in points]
        l.append(sub_list)

clustering = pd.DataFrame(l)
clustering = clustering.sample(frac = 1)
clustering = clustering.rename(columns = {4:"id"})
clustering = clustering.rename(columns = {0:"kmeans_cluster"})
clustering = clustering[["id","kmeans_cluster"]]

virdi = pd.merge(virdi, pd.DataFrame(clustering), on = "id")

pd.DataFrame.to_csv(virdi, "C:/Users/Tobias/data/virdi_with_clustering.csv")
time.sleep(10)
"""