# -*- coding: utf-8 -*-

from scipy.spatial.distance import pdist, squareform
import time, datetime
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

index = 10
print "Index: " + str(index)

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

district_input = ""
kmeans_bool = False
admin_bool = False
while district_input not in ["k","a","n"]:
    district_input = raw_input("(k)-means, (a)dmin or (n)o district variables? ")

if district_input == "k":
    kmeans_bool = True
    KMEANS_K = int(raw_input("K: "))
    print "Running K-means with k = " + str(KMEANS_K)
    district_string = "K-means"
elif district_input == "a":
    admin_bool = True
    print "Running admin districts"
else:
    district_string = "Districtless"
    print "Running without district variables"

data_string = "kriging_and_gwr_results_"
if kriging_bool:
    if kmeans_bool:
        data_string = "kriging_kmeans_results_"
    elif admin_bool:
        data_string = "kriging_admin_results_"
    else:
        data_string = "kriging_districtless_results_"

def get_quarter(date):
    if pd.isnull(date):
        return np.nan
    year = date.year
    month = date.month
    quarter = 1 + ((month - 1) // 3)
    quarter_string = str(year) + "_Q" + str(quarter)
    return quarter_string


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
    elif admin_bool:
        r_training = pd.read_csv("C:/Users/" + USER_STRING + "/data/training_r_admin_" + str(index) + ".csv")
    else:
        r_training = pd.read_csv("C:/Users/" + USER_STRING + "/data/training_r_districtless_" + str(index) + ".csv")

else:
    test = test.assign(reg_prediction = r_test.gwr_pred)
    if kmeans_bool:
        r_training = pd.read_csv("C:/Users/" + USER_STRING + "/data/training_r_kmeans_gwr_" + str(index) + ".csv")
        merge_cols.append("kmeans_cluster")
    else:
        r_training = pd.read_csv("C:/Users/" + USER_STRING + "/data/training_r_admin_gwr_" + str(index) + ".csv")
    r_training = r_training.assign(regression_residuals = r_training.gwr_residual)

py_training = pd.read_csv("C:/Users/" + USER_STRING + "/data/training" + str(index) + ".csv")

if kmeans_bool:
    py_training = py_training[["id", "kmeans_cluster", "real_sold_date"]]
else:
    py_training = py_training[["id", "real_sold_date"]]

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

print "Constructing basic estimates for test set"

# ------------
# Construct basic estimates for test set
# ------------

number_of_prev_sales_dummy = pd.get_dummies(test.number_of_prev_sales)

basic_est_0 = test.reg_prediction_nat
basic_est_1 = REG_SHARE * test.reg_prediction_nat + (1 - REG_SHARE) * test.prev_sale_estimate_1
basic_est_2 = REG_SHARE * test.reg_prediction_nat + (1 - REG_SHARE) * (0.9 * test.prev_sale_estimate_1 + 0.1 * test.prev_sale_estimate_2)
basic_est_3 = REG_SHARE * test.reg_prediction_nat + (1 - REG_SHARE) * (0.85 * test.prev_sale_estimate_1 + 0.15 * (0.8 * test.prev_sale_estimate_2 + 0.2 * test.prev_sale_estimate_3))

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

################


csv_results = pd.DataFrame(columns = ["Ordinary" + model_string, "Repeat Combination", "Time"])
ordinary_results = pd.Series()
repeat_results = pd.Series()

print ""
print " -------- Test Results -------- "
print ""
print "Regression, median error:", 100 * round(score.reg_deviation_percentage.median(), 5), "%"
print score.reg_deviation_percentage.quantile([.25, .5, .75])

ordinary_results = ordinary_results.append(score.reg_deviation_percentage.quantile([.25, .5, .75]), ignore_index=True)

w10 = score.reg_deviation_percentage[score.reg_deviation_percentage <= 0.1].count() / np.float(
        score.reg_deviation_percentage.size)

ordinary_results = ordinary_results.append(pd.Series(w10), ignore_index=True)

print "Within 10%: " + str(round(100 * w10, 2)) + "%"
if moran_and_geary_bool:
    moran, geary = spatial_measures.get_i_and_c(test, MORANS_SET_SIZE, "regression_residual")
    # regression_residual
    print ""
    print "Moran's I: " + str(100 * round(moran, 4)) + "% and Geary's C: " + str(100 * round(geary, 4)) + "%."

ordinary_results = ordinary_results.append(pd.Series([moran, geary]), ignore_index=True)

print ""
print "Repeat 1, median error:", 100 * round(score.rep_1_deviation_percentage.median(), 5), "%"
# print score.rep_1_deviation_percentage.quantile([.25, .5, .75])
print "Repeat 2, median error:", 100 * round(score.rep_2_deviation_percentage.median(), 5), "%"
# print score.rep_2_deviation_percentage.quantile([.25, .5, .75])
print "Repeat 3, median error:", 100 * round(score.rep_3_deviation_percentage.median(), 5), "%"
# print score.rep_3_deviation_percentage.quantile([.25, .5, .75])


print ""
print "Estimate adjusted with repeated sales, median error:", 100 * round(score.basic_deviation_percentage.median(),
                                                                          5), "%"
print score.basic_deviation_percentage.quantile([.25, .5, .75])

repeat_results = repeat_results.append(score.basic_deviation_percentage.quantile([.25, .5, .75]), ignore_index=True)

w10 = score.basic_deviation_percentage[score.basic_deviation_percentage <= 0.1].count() / np.float(
        score.basic_deviation_percentage.size)

repeat_results = repeat_results.append(pd.Series(w10), ignore_index=True)

print "Within 10%: " + str(round(100 * w10, 2)) + "%"
if moran_and_geary_bool:
    moran, geary = spatial_measures.get_i_and_c(test, MORANS_SET_SIZE, "basic_estimate_residual")
    # basic_estimate_residual
    print ""
    print "Moran's I: " + str(100 * round(moran, 4)) + "% and Geary's C: " + str(100 * round(geary, 4)) + "%."

repeat_results = repeat_results.append(pd.Series([moran, geary]), ignore_index=True)

csv_results["Ordinary" + model_string] = ordinary_results
csv_results["Repeat Combination"] = repeat_results
csv_results["Time"]  = pd.Series(datetime.datetime.now().strftime ("%c"))

csv_file_name = "C:/Users/" + USER_STRING + "/data/" + model_string

if kmeans_bool:
    csv_file_name += "_kmeans_results.csv"
elif admin_bool:
    print "Used administrative districts"
    csv_file_name += "_admin_results.csv"
else:
    print "Used districtless"
    csv_file_name += "_districtless_results.csv"
print "Reg share: " + str(REG_SHARE)


header_bool = False
try:
    f = open(csv_file_name, 'r')
except IOError:
    header_bool = True
csv_results = csv_results.append(pd.Series(["","",""]),ignore_index=True)

with open(csv_file_name, 'a') as f:
    csv_results.to_csv(f, header=header_bool)
    print "Written to file"