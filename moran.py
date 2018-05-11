# -*- coding: utf-8 -*-

from scipy.spatial.distance import pdist, squareform
import time
import alva_io, address
import pandas as pd
import numpy as np
import calendar
from patsy import dmatrices
import statsmodels.api as sm
import matplotlib.pyplot as plt


def i(test, MORANS_SET_SIZE, residual_column_name):

    print "Calculating Moran's I and Geary's C"

    residuals = test[[residual_column_name]]

    print "0%",
    nevner, teller_morans, teller_geary = 0, 0, 0

    #comparable_matrix = test.loc[abs(comparable_matrix.coord_x - r.coord_x) + abs(comparable_matrix.coord_y - r.coord_y) < 0.02]
    distance_matrix = squareform(pdist(test[["coord_x", "coord_y"]]))

    morans_weights_dict = dict()

    count = 0.0
    size = float(len(test.index))
    old_progress, new_progress = 0,0

    for index, r in test.iterrows():

        close_ids = []
        close_ids_test = []
        distance_array = distance_matrix[index]

        max_distance = 0.000025
        prev_max_distance = -1

        while len(close_ids) < MORANS_SET_SIZE + 1:
            close_points = (distance_array <= max_distance) & (distance_array > prev_max_distance)
            if max_distance == 0.000025:
                close_points[index] = False  # exclude itself
            close_ids_test += [test.iloc[i].name for i, close in enumerate(close_points) if close]
            close_ids += [i for i, close in enumerate(close_points) if close]
            prev_max_distance = max_distance
            max_distance *= 1.5

        close_ids_test = close_ids_test[:MORANS_SET_SIZE]
        close_ids = close_ids[:MORANS_SET_SIZE]
        close_residuals = residuals.loc[close_ids_test]
        own_residual = residuals.loc[index][0]

        distances = distance_array[[close_ids]]

        if len(distances) > 0:
            if distances.max() == 0:
                distances[distances == 0] = 0.000025 # value is irrelevant
            else:
                distances[distances == 0] = distances[distances > 0].min()

        distances = 1 / distances

        morans_weights = pd.Series(distances / distances.sum())

        close_residuals = close_residuals.reset_index(drop = True)
        close_residuals = close_residuals[residual_column_name]

        teller_morans += (morans_weights * close_residuals * own_residual).sum()
        teller_geary += (morans_weights * (close_residuals - own_residual)**2).sum()
        nevner += own_residual ** 2

        new_progress = round(count / size,2)
        if old_progress != new_progress:
            if (int(100*new_progress)) % 10 == 0:
                print str(int(100*new_progress)) + "%",
            else:
                print "|",
        old_progress = new_progress

        # print count
        count += 1

    print ""
    return teller_morans / nevner, (len(test.index) - 1) * teller_geary / (nevner * 2 * len(test.index))