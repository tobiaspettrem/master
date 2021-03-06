
from pylab import *
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import random
import operator

def plot_districts(data_set, district_column_name, PRICE_FACTOR, RUN_NUMBER): #districts is a set of unique districts
    price_dict = dict()

    district_type = "K-means"
    if district_column_name == "bydel_code":
        district_type = "Admin"

    districts = list(set( data_set[[district_column_name]].iloc[:,0]))

    number_of_districts = len(districts)

    # map pricing of districts to get ranking
    for i in range(number_of_districts):
        district = districts[i]
        average_price = data_set.loc[data_set[district_column_name] == district, "kr/m2"].mean()
        price_dict[district] = average_price
        print "District: " + str(district) + ", average sqm price: " + str(round(average_price))

    sorted_districts = sorted(price_dict.items(), key=operator.itemgetter(1))

    color_ranking = pd.DataFrame(columns=[district_column_name, "ranking"])
    for i in range(number_of_districts):
        color_ranking = color_ranking.append(pd.DataFrame([[sorted_districts[i][0], number_of_districts - i]], columns=[district_column_name, "ranking"]))

    data_set = pd.merge(data_set, color_ranking, how = "left", on = district_column_name)

    print color_ranking

    random.shuffle(districts)

    cmap = 'tab20'

    hot = plt.get_cmap(cmap)
    cNorm  = colors.Normalize(vmin=0, vmax=number_of_districts)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=hot)
    #plt.ion()

    # Plot each species
    #for i in range(number_of_districts):
    #    index = data_set[[district_column_name]].iloc[:,0] == districts[i]
    #    plt.scatter(data_set.coord_x[index], data_set.coord_y[index], s=10, color=scalarMap.to_rgba(i), label=districts[i])

    plt.scatter(data_set.coord_x, data_set.coord_y, s=10, c = data_set.ranking, cmap=plt.cm.RdYlGn)
    plt.title(str(number_of_districts) + district_type + " districts. Price Fact.: " + str(PRICE_FACTOR) + ". Cmap: " + cmap + ". Run no.: " + str(RUN_NUMBER))
    plt.show()
    plt.pause(0.2)
    print "Updating plot"


#db_link = 'C:/Users/tobiasrp/data/training2.csv'
#virdi = pd.read_csv(db_link)
#virdi = virdi[["kr/m2", "coord_x", "coord_y", "bydel_code", "kmeans_cluster"]]


#print "Plotting districts"
#plot_districts(virdi, "bydel_code", 0, 2)
#plot_districts(virdi, "kmeans_cluster", 3, 2)


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

