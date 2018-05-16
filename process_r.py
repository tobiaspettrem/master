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
COMPARABLE_SET_SIZE = 15
AUTOREGRESSIVE_COMPARABLE_SET_SIZE = 15
MORANS_SET_SIZE = 15
RUN_LAD = True
REG_SHARE = 0.5
KMEANS_K = 14
KNN_K = 5


def get_quarter(date):
    if pd.isnull(date):
        return np.nan
    year = date.year
    month = date.month
    quarter = 1 + ((month - 1) // 3)
    quarter_string = str(year) + "_Q" + str(quarter)
    return quarter_string

index = 5

test = pd.read_csv("C:/Users/" + USER_STRING + "/data/kriging_admin_results_" + str(index) + ".csv")

print "Kriging Admin " + str(index)


test = test.assign(reg_prediction = test.kriging_pred)
#test = test.assign(reg_prediction = test.gwr_pred)


print "Constructing basic estimates for test set"

# ------------
# Construct basic estimates for test set
# ------------

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

number_of_prev_sales_dummy = pd.get_dummies(test.number_of_prev_sales)

basic_est_0 = test.reg_prediction_nat
basic_est_1 = REG_SHARE * test.reg_prediction_nat + (1 - REG_SHARE) * test.prev_sale_estimate_1
basic_est_2 = REG_SHARE * test.reg_prediction_nat + (1 - REG_SHARE) * (0.9 * test.prev_sale_estimate_1 + 0.1 * test.prev_sale_estimate_2)
basic_est_3 = REG_SHARE * test.reg_prediction_nat + (1 - REG_SHARE) * (0.85 * test.prev_sale_estimate_1 + 0.15 * (0.8 * test.prev_sale_estimate_2 + 0.2 * test.prev_sale_estimate_3))

basic_estimate_matrix = pd.concat([basic_est_0,basic_est_1,basic_est_2,basic_est_3], axis = 1)

basic_estimate_matrix = pd.DataFrame(data=np.where(number_of_prev_sales_dummy, basic_estimate_matrix, 0))
basic_estimate = basic_estimate_matrix.sum(axis = 1)
basic_estimate = basic_estimate.astype(int)

test = test.reset_index()

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


print ""
print " -------- Test Results -------- "
print ""
print "Medianfeil regresjon:",100*round(score.reg_deviation_percentage.median(),5),"%"
moran, geary = spatial_measures.get_i_and_c(test, 100, "regression_residual")
#regression_residual
print ""
print "Moran's I: " + str(round(moran, 4)) + " and Geary's C: " + str(round(geary, 4))
print "Within 10%: " + str(round(100 * score.reg_deviation_percentage[score.reg_deviation_percentage <= 0.1].count() / np.float(score.reg_deviation_percentage.size), 2)) + "%"
print ""
print "Gjennomsnittsfeil regresjon:",100*round(score.reg_deviation_percentage.mean(),5),"%"
print score.reg_deviation_percentage.quantile([.25, .5, .75])

print ""
print "Repeat 1, medianfeil:",100*round(score.rep_1_deviation_percentage.median(),5),"%"
print score.rep_1_deviation_percentage.quantile([.25, .5, .75])
print "Repeat 2, medianfeil:",100*round(score.rep_2_deviation_percentage.median(),5),"%"
print score.rep_2_deviation_percentage.quantile([.25, .5, .75])
print "Repeat 3, medianfeil:",100*round(score.rep_3_deviation_percentage.median(),5),"%"
print score.rep_3_deviation_percentage.quantile([.25, .5, .75])

print ""
print "Basic-estimat, medianfeil:",100*round(score.basic_deviation_percentage.median(),5),"%"
print score.basic_deviation_percentage.quantile([.25, .5, .75])
moran, geary = spatial_measures.get_i_and_c(test, 100, "basic_estimate_residual")
print ""
print "Moran's I: " + str(round(moran, 4)) + " and Geary's C: " + str(round(geary, 4))
print "Within 10%: " + str(round(100 * score.basic_deviation_percentage[score.basic_deviation_percentage <= 0.1].count() / np.float(score.basic_deviation_percentage.size), 2)) + "%"
#basic_estimate_residual
"""
print "Comparable-estimat, medianfeil:",100*round(score.comparable_deviation_percentage.median(),5),"%"
print score.comparable_deviation_percentage.quantile([.25, .5, .75])
moran, geary = spatial_measures.get_i_and_c(test, 100, "post_resid_adjustment_residual")
print ""
print "Moran's I: " + str(round(moran, 4)) + " and Geary's C: " + str(round(geary, 4))
print "Within 10%: " + str(round(100 * score.comparable_deviation_percentage[score.comparable_deviation_percentage <= 0.1].count() / np.float(score.comparable_deviation_percentage.size), 2)) + "%"
#post_resid_adjustment_residual

"""