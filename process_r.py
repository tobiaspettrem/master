# -*- coding: utf-8 -*-

from scipy.spatial.distance import pdist, squareform
import time
import spatial_measures
import alva_io, address
import pandas as pd
import numpy as np
import calendar
from patsy import dmatrices
import statsmodels.api as sm
import kmeans
import kmeans_district_map
import pysal
import matplotlib.pyplot as plt

USER_STRING = "tobiasrp"
SOLD_TO_OFFICIAL_DIFF = 55
TRAINING_SET_SIZE = 0.8
COMPARABLE_SET_SIZE = 6
AUTOREGRESSIVE_COMPARABLE_SET_SIZE = 15
MORANS_SET_SIZE = 15
RUN_LAD = True
REG_SHARE = 0.6
KMEANS_K = 14
KNN_K = 5

model_string = "GWR"
district_string = "Admin"

moran_and_geary_bool = ""
while moran_and_geary_bool not in ["y","n"]:
    moran_and_geary_bool = raw_input("Calculate Geary's C and Moran's I? y or n: ")

if moran_and_geary_bool == "y":
    moran_and_geary_bool = True
else:
    moran_and_geary_bool = False

kriging_bool = ""
while kriging_bool not in ["y","n"]:
    kriging_bool = raw_input("Kriging? y or n: ")

if kriging_bool == "y":
    kriging_bool = True
    model_string = "Kriging"
else:
    kriging_bool = False

kmeans_bool = ""
while kmeans_bool not in ["y","n"]:
    kmeans_bool = raw_input("Results with k-means? y or n: ")

if kmeans_bool == "y":
    kmeans_bool = True
    district_string = "K-means"
else:
    kmeans_bool = False

data_string = "kriging_and_gwr_results_"
if kriging_bool:
    data_string = "kriging_admin_results_"
    if kmeans_bool:
        data_string = "kriging_kmeans_results_"

def get_quarter(date):
    if pd.isnull(date):
        return np.nan
    year = date.year
    month = date.month
    quarter = 1 + ((month - 1) // 3)
    quarter_string = str(year) + "_Q" + str(quarter)
    return quarter_string

index = 5

print "Model: " + model_string + ", district: " + district_string + ", run number: " + str(index)

py_test = pd.read_csv("C:/Users/" + USER_STRING + "/data/test" + str(index) + ".csv")

py_test = py_test[["id", "kmeans_cluster_prediction", "real_sold_date"]]

r_test = pd.read_csv("C:/Users/" + USER_STRING + "/data/" + data_string + str(index) + ".csv")

merge_cols = ["id"]

test = pd.merge(r_test,py_test,on=merge_cols)

test = test.drop(columns = ["reg_prediction"])

if kriging_bool:
    test = test.assign(reg_prediction = r_test.kriging_pred)
    if kmeans_bool:
        r_training = pd.read_csv("C:/Users/" + USER_STRING + "/data/training_r_kmeans_" + str(index) + ".csv")
        merge_cols.append("kmeans_cluster")
    else:
        r_training = pd.read_csv("C:/Users/" + USER_STRING + "/data/training_r_admin_" + str(index) + ".csv")

else:
    test = test.assign(reg_prediction = r_test.gwr_pred)
    if kmeans_bool:
        r_training = pd.read_csv("C:/Users/" + USER_STRING + "/data/training_r_kmeans_gwr_" + str(index) + ".csv")
        merge_cols.append("kmeans_cluster")
    else:
        r_training = pd.read_csv("C:/Users/" + USER_STRING + "/data/training_r_admin_gwr_" + str(index) + ".csv")
    r_training = r_training.assign(regression_residuals = r_training.gwr_residual)

py_training = pd.read_csv("C:/Users/" + USER_STRING + "/data/training" + str(index) + ".csv")

py_training = py_training[["id", "kmeans_cluster", "real_sold_date"]]

training = pd.merge(py_training,r_training,on=merge_cols)


# map previous sales
test = test.assign(reg_prediction_nat = (np.exp(test.reg_prediction) * test.prom).astype(int))
test = test.assign(regression_residual = test.log_price_plus_comdebt - test.reg_prediction)

### Remove repeated estimates if it deviates too much from fitted value (test set)

prev_sale_price_1_reasonable = (abs(test.prev_sale_estimate_1 - test.reg_prediction_nat) / test.reg_prediction_nat) < 0.25 ### using method from previous model
prev_sale_price_2_reasonable = (abs(test.prev_sale_estimate_2 - test.reg_prediction_nat) / test.reg_prediction_nat) < 0.25 ### using method from previous model
prev_sale_price_3_reasonable = (abs(test.prev_sale_estimate_3 - test.reg_prediction_nat) / test.reg_prediction_nat) < 0.25 ### using method from previous model

test = test.assign(prev_sale_estimate_3 = np.where(prev_sale_price_3_reasonable, test.prev_sale_estimate_3,np.nan))                             # set 3 to nan if invalid
test = test.assign(prev_sale_estimate_2 = np.where(prev_sale_price_2_reasonable, test.prev_sale_estimate_2,test.prev_sale_estimate_3))          # set 2 to 3 if 2 invalid
test = test.assign(prev_sale_estimate_3 = np.where(prev_sale_price_2_reasonable, test.prev_sale_estimate_3,np.nan))                             # remove 3 if 2 invalid
test = test.assign(prev_sale_estimate_1 = np.where(prev_sale_price_1_reasonable, test.prev_sale_estimate_1,test.prev_sale_estimate_2))          # set 1 to 2 if 1 invalid
test = test.assign(prev_sale_estimate_2 = np.where(prev_sale_price_1_reasonable, test.prev_sale_estimate_2,np.nan))                             # remove 2 if 1 invalid

prev_sale_price_2_reasonable = (abs(test.prev_sale_estimate_2 - test.reg_prediction_nat) / test.reg_prediction_nat) < 0.25 ### using method from previous model
test = test.assign(prev_sale_estimate_2 = np.where(prev_sale_price_2_reasonable, test.prev_sale_estimate_2,test.prev_sale_estimate_3))          # set 2 to 3 if 2 invalid
test = test.assign(prev_sale_estimate_3 = np.where(prev_sale_price_2_reasonable, test.prev_sale_estimate_3,np.nan))                             # remove 3 if 2 invalid

prev_sale_price_1_exists = ~pd.isnull(test.prev_sale_estimate_1)
prev_sale_price_2_exists = ~pd.isnull(test.prev_sale_estimate_2)
prev_sale_price_3_exists = ~pd.isnull(test.prev_sale_estimate_3)

number_of_prev_sales = prev_sale_price_1_exists.astype(int) + prev_sale_price_2_exists.astype(int) + prev_sale_price_3_exists.astype(int)

test = test.assign(number_of_prev_sales = number_of_prev_sales)

# mapping done


# ----------------
# Simple comparable implementation
# ----------------

test = test.reset_index()
training = training.reset_index()

count = 0.0
size = float(len(test.index))
old_progress, new_progress = 0,0
resid_median_list = []
number_of_close_resids_list = []
close_resids_matrix = []
number_below_three = 0

print ""
print "Mapping distances for comparable method"
print "0%",

mapping_start = time.time()

for index, r in test.iterrows():
    if kriging_bool:
        if kmeans_bool:
            district_matrix = test.loc[test.kmeans_cluster_prediction == r.kmeans_cluster_prediction] #HUSK Å NOTERE DETTE ET STED
        else:
            district_matrix = test.loc[test.bydel_code == r.bydel_code] #HUSK Å NOTERE DETTE ET STED
    else:
        if kmeans_bool:
            district_matrix = training.loc[training.kmeans_cluster == r.kmeans_cluster_prediction] #HUSK Å NOTERE DETTE ET STED
        else:
            district_matrix = training.loc[training.bydel_code == r.bydel_code] #HUSK Å NOTERE DETTE ET STED

    district_matrix = district_matrix.loc[district_matrix.real_sold_date < r.real_sold_date] # comparable sale must have occured earlier in time

    district_matrix = district_matrix.append(r)

    comparable_coords = district_matrix[["coord_x", "coord_y"]].as_matrix()
    N = len(comparable_coords)
    distance_matrix = np.zeros(N)
    loni, lati = comparable_coords[-1]
    for j in xrange(N):
        lonj, latj = comparable_coords[j]
        distance_matrix[j] = kmeans.coord_distance(lati, loni, latj, lonj)

    max_distance = 0.001  # in km, so 1 meter
    prev_max_distance = -1
    close_indexes = []

    while len(close_indexes) < COMPARABLE_SET_SIZE and max_distance <= 0.15:
        close_points = (distance_matrix <= max_distance) & (distance_matrix > prev_max_distance)
        if max_distance == 0.001:
            close_points[-1] = False  # exclude itself
        close_indexes += [index for index, close in enumerate(close_points) if close]
        prev_max_distance = max_distance
        max_distance *= 1.1

    close_indexes = close_indexes[:COMPARABLE_SET_SIZE]

    close_ids = [district_matrix.iloc[index].name for index in close_indexes]

    """
    #OLD METHOD
    distance_matrix = squareform(pdist(district_matrix[["coord_x","coord_y"]]))

    max_distance = 0.000025
    prev_max_distance = -1
    close_ids = []

    row_num_prev_sales = r.number_of_prev_sales

    while len(close_ids) < COMPARABLE_SET_SIZE and max_distance <= 0.0025:
        close_points = (distance_matrix[-1] <= max_distance) & (distance_matrix[-1] > prev_max_distance)
        if max_distance == 0.000025:
            close_points[-1] = False # exclude itself
        close_ids += [district_matrix.iloc[index].name for index, close in enumerate(close_points) if close]
        prev_max_distance = max_distance
        max_distance *= 1.1

    close_ids = close_ids[:COMPARABLE_SET_SIZE]
    """

    if kriging_bool:
        close_resids = test.loc[close_ids].kriging_residual
    else:
        close_resids = training.loc[close_ids].regression_residuals

    close_resids_matrix.append(list(close_resids))

    resid_median = close_resids.median() * 0.7

    number_of_close_resids = len(close_resids)
    number_of_close_resids_list.append(number_of_close_resids)

    if number_of_close_resids < 3: # if the property to be estimated has few neighboring residuals, then put less emphasis on them
        number_below_three += 1
        resid_median *= 0.5

    resid_median_list.append(resid_median)

    new_progress = round(count / size,2)
    if old_progress != new_progress:
        if (int(100*new_progress)) % 10 == 0:
            print str(int(100*new_progress)) + "%",
        else:
            print "|",
    old_progress = new_progress

    count += 1

print ""
print "Done mapping distances. " + str(round(time.time() - mapping_start)) + " seconds elapsed."
print str(number_below_three) + " dwellings out of " + str(len(test.index)) + " had fewer than 3 neighbors (" + str(100 * round(float(number_below_three) / len(test.index), 4)) +"%)"

resid_median_column = pd.Series(resid_median_list)
number_of_close_resids_column = pd.Series(number_of_close_resids_list)

test = test.assign(number_of_close_resids = number_of_close_resids_column)

test = test.assign(comparable_estimate = test.reg_prediction + resid_median_column)
test = test.assign(comparable_estimate = test.comparable_estimate.fillna(test.reg_prediction))

test = test.assign(comparable_estimate_nat = (test.prom * np.exp(test.comparable_estimate)).astype(int))

test = test.assign(resid_median_column = resid_median_column)
test = test.assign(pre_resid_adjustment_residual = test.log_price_plus_comdebt - test.reg_prediction)
test = test.assign(post_resid_adjustment_residual = test.log_price_plus_comdebt - test.comparable_estimate)

close_resids_df = pd.DataFrame(close_resids_matrix)

alva_io.write_to_csv(test,"C:/Users/" + USER_STRING + "/data/residual_adjusted_estimates.csv")
alva_io.write_to_csv(close_resids_df,"C:/Users/" + USER_STRING + "/data/close_resids_df.csv")

# END COMPARABLE MODEL









print "Constructing basic estimates for test set"

# ------------
# Construct basic estimates for test set
# ------------

number_of_prev_sales_dummy = pd.get_dummies(test.number_of_prev_sales)

basic_est_0 = test.reg_prediction_nat
basic_est_1 = REG_SHARE * test.comparable_estimate_nat + (1 - REG_SHARE) * test.prev_sale_estimate_1
basic_est_2 = REG_SHARE * test.comparable_estimate_nat + (1 - REG_SHARE) * (0.9 * test.prev_sale_estimate_1 + 0.1 * test.prev_sale_estimate_2)
basic_est_3 = REG_SHARE * test.comparable_estimate_nat + (1 - REG_SHARE) * (0.85 * test.prev_sale_estimate_1 + 0.15 * (0.8 * test.prev_sale_estimate_2 + 0.2 * test.prev_sale_estimate_3))

basic_estimate_matrix = pd.concat([basic_est_0,basic_est_1,basic_est_2,basic_est_3], axis = 1)

basic_estimate_matrix = pd.DataFrame(data=np.where(number_of_prev_sales_dummy, basic_estimate_matrix, 0))
basic_estimate = basic_estimate_matrix.sum(axis = 1)
basic_estimate = basic_estimate.astype(int)

basic_estimate_log = np.log(basic_estimate / test.prom)
basic_estimate_residual = test.log_price_plus_comdebt - basic_estimate_log

test = test.assign(basic_estimate = basic_estimate)
test = test.assign(basic_estimate_log = basic_estimate_log)
test = test.assign(basic_estimate_residual = basic_estimate_residual)

#################################



score = pd.DataFrame()
score = score.assign(true_value = np.exp(test["log_price_plus_comdebt"]) * test.prom)

score = score.assign(reg_prediction = test.reg_prediction_nat)
score = score.assign(reg_deviation = score.true_value - score.reg_prediction)
score = score.assign(reg_deviation_percentage = abs(score.reg_deviation / score.true_value))
score = score.assign(rep_1_prediction = test.prev_sale_estimate_1)
score = score.assign(rep_2_prediction = test.prev_sale_estimate_2)
score = score.assign(rep_3_prediction = test.prev_sale_estimate_3)
score = score.assign(rep_1_deviation = score.true_value - score.rep_1_prediction)
score = score.assign(rep_2_deviation = score.true_value - score.rep_2_prediction)
score = score.assign(rep_3_deviation = score.true_value - score.rep_3_prediction)
score = score.assign(rep_1_deviation_percentage = abs(score.rep_1_deviation / score.true_value))
score = score.assign(rep_2_deviation_percentage = abs(score.rep_2_deviation / score.true_value))
score = score.assign(rep_3_deviation_percentage = abs(score.rep_3_deviation / score.true_value))

score = score.assign(basic_prediction = test.basic_estimate)
score = score.assign(basic_deviation = score.true_value - score.basic_prediction)
score = score.assign(basic_deviation_percentage = abs(score.basic_deviation / score.true_value))

score = score.assign(comparable_prediction = test.comparable_estimate_nat)
score = score.assign(comparable_deviation = score.true_value - score.comparable_prediction)
score = score.assign(comparable_deviation_percentage = abs(score.comparable_deviation / score.true_value))

print ""
print " -------- Test Results -------- "
print ""
print "Regression, median error:",100*round(score.reg_deviation_percentage.median(),5),"%"
print score.reg_deviation_percentage.quantile([.25, .5, .75])
print "Within 10%: " + str(round(100 * score.reg_deviation_percentage[score.reg_deviation_percentage <= 0.1].count() / np.float(score.reg_deviation_percentage.size), 2)) + "%"
if moran_and_geary_bool:
    moran, geary = spatial_measures.get_i_and_c(test, 100, "regression_residual")
    #regression_residual
    print ""
    print "Moran's I: " + str(100 * round(moran, 4)) + "% and Geary's C: " + str(100 * round(geary, 4)) + "%."
print ""
print "Average error, regression:",100*round(score.reg_deviation_percentage.mean(),5),"%"

print ""
print "Repeat 1, median error:",100*round(score.rep_1_deviation_percentage.median(),5),"%"
#print score.rep_1_deviation_percentage.quantile([.25, .5, .75])
print "Repeat 2, median error:",100*round(score.rep_2_deviation_percentage.median(),5),"%"
#print score.rep_2_deviation_percentage.quantile([.25, .5, .75])
print "Repeat 3, median error:",100*round(score.rep_3_deviation_percentage.median(),5),"%"
#print score.rep_3_deviation_percentage.quantile([.25, .5, .75])

print ""
print "Comparable-estimate, median error:",100*round(score.comparable_deviation_percentage.median(),5),"%"
print score.comparable_deviation_percentage.quantile([.25, .5, .75])
print "Within 10%: " + str(round(100 * score.comparable_deviation_percentage[score.comparable_deviation_percentage <= 0.1].count() / np.float(score.comparable_deviation_percentage.size), 2)) + "%"
if moran_and_geary_bool:
    moran, geary = spatial_measures.get_i_and_c(test, 100, "post_resid_adjustment_residual")
    print ""
    print "Moran's I: " + str(100 * round(moran, 4)) + "% and Geary's C: " + str(100 * round(geary, 4)) + "%."
#post_resid_adjustment_residual

print ""
print "Estimate adjusted with repeated sales, median error:",100*round(score.basic_deviation_percentage.median(),5),"%"
print score.basic_deviation_percentage.quantile([.25, .5, .75])
print "Within 10%: " + str(round(100 * score.basic_deviation_percentage[score.basic_deviation_percentage <= 0.1].count() / np.float(score.basic_deviation_percentage.size), 2)) + "%"
if moran_and_geary_bool:
    moran, geary = spatial_measures.get_i_and_c(test, 100, "basic_estimate_residual")
    print ""
    print "Moran's I: " + str(100 * round(moran, 4)) + "% and Geary's C: " + str(100 * round(geary, 4)) + "%."
#basic_estimate_residual

if kmeans_bool:
    print "Number of k-means districts was: " + str(KMEANS_K)
else:
    print "Used administrative districts"
print "Reg share: " + str(REG_SHARE)