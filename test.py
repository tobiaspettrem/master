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
from math import cos, asin, sqrt
import kmeans_district_map
import random
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
RUN_NUMBER = -1

print RUN_NUMBER

if RUN_LAD:
    print "Running LAD with basic model regression share: " + str(REG_SHARE * 100) + "%."
else:
    print "Running OLS with basic model regression share: " + str(REG_SHARE * 100) + "%."


def get_quarter(date):
    if pd.isnull(date):
        return np.nan
    year = date.year
    month = date.month
    quarter = 1 + ((month - 1) // 3)
    quarter_string = str(year) + "_Q" + str(quarter)
    return quarter_string


virdi_augmented = pd.read_csv("C:/Users/" + USER_STRING + "/data/virdi_aug_title_prev_sale_correct.csv", sep=";")
virdi_augmented = virdi_augmented.assign(real_sold_date=pd.to_datetime(virdi_augmented["real_sold_date"], format="%d.%m.%Y"))

print virdi_augmented.shape
wp_bool = False
kmeans_bool = False
moran_and_geary_bool = True

# virdi = virdi.loc[virdi["kr/m2"] > ] # cutting on sqm price

# MANUALLY CORRECTING SOME VERY WRONG PRICES. BASED ON FINN AND EIENDOMSVERDI

virdi_augmented.loc[virdi_augmented.ad_code == 85468204.0, "Finalprice"] = 3290000.0
virdi_augmented.loc[virdi_augmented.ad_code == 85468204.0, "Total price"] = 3290000.0 + 41000.0
virdi_augmented.loc[virdi_augmented.ad_code == 85468204.0, "kr/m2"] = 48985.2941

virdi_augmented.loc[virdi_augmented.ad_code == 98106987, "Finalprice"] = 4325000.0
virdi_augmented.loc[virdi_augmented.ad_code == 98106987, "Total price"] = 4325000.0 + 288000.0
virdi_augmented.loc[virdi_augmented.ad_code == 98106987, "kr/m2"] = 78186.4407

virdi_augmented.loc[virdi_augmented.ad_code == 91039330, "Finalprice"] = 3100000.0
virdi_augmented.loc[virdi_augmented.ad_code == 91039330, "Total price"] = 3100000.0 + 42107.0
virdi_augmented.loc[virdi_augmented.ad_code == 91039330, "kr/m2"] = 48340.0

virdi_augmented.loc[virdi_augmented.ad_code == 104814522, "Finalprice"] = 3045000.0
virdi_augmented.loc[virdi_augmented.ad_code == 104814522, "Total price"] = 3045000.0 + 104160.0
virdi_augmented.loc[virdi_augmented.ad_code == 104814522, "kr/m2"] = 54296.0

virdi_augmented = virdi_augmented.assign(log_price_plus_comdebt=np.log(virdi_augmented["kr/m2"]))

# ----- CREATE NEW COLUMN size_group ------ #

size_group_size = 10  # change this only to change group sizes
size_labels = ["10 -- 29"]
size_labels += ["{0} -- {1}".format(i, i + size_group_size - 1) for i in range(30, 150, size_group_size)]
size_labels.append("150 -- 179")
size_labels.append("180 -- ")

size_thresholds = [10]
size_thresholds += [i for i in range(30, 151, size_group_size)] + [180] + [500]

virdi_augmented['size_group'] = pd.cut(virdi_augmented.prom, size_thresholds, right=False, labels=size_labels)

# ----- CREATE NEW COLUMN sold_month ------#

# add sold_month column for discovery of seasonal effects

virdi_augmented = virdi_augmented.assign(sold_month=virdi_augmented.real_sold_date.map(lambda x: x.month))
virdi_augmented = virdi_augmented.assign(sold_year=virdi_augmented.real_sold_date.map(lambda x: x.year))
virdi_augmented = virdi_augmented.assign(
    sold_month=virdi_augmented.sold_month.apply(lambda x: calendar.month_abbr[x]))  # text instead of int
virdi_augmented = virdi_augmented.assign(
    sold_month_and_year=virdi_augmented.sold_month + "_" + virdi_augmented.sold_year.map(str))

# delete sold_date and official_date now that real_sold_date covers both
del virdi_augmented["sold_date"]
del virdi_augmented["official_date"]
del virdi_augmented["register_date"]

# ------------------
#   DO REGRESSION
# ------------------

# virdi_augmented = virdi_augmented[["log_price_plus_comdebt","size_group","sold_month_and_year","bydel_code","unit_type", "prom","Total price","coord_x","coord_y"]]

columns_to_count = ["size_group", "sold_month_and_year", "bydel_code", "unit_type"]

# virdi_augmented = virdi_augmented.loc[virdi_augmented.bydel_code != "SEN"]

virdi_augmented.loc[virdi_augmented.bydel_code == 'SEN', 'bydel_code'] = "bsh"  # reassign SENTRUM to St. Hanshaugen

virdi_augmented = virdi_augmented.loc[virdi_augmented.bydel_code != "MAR"]

virdi_augmented = virdi_augmented.loc[virdi_augmented.sold_month_and_year != "Feb_2018"]
virdi_augmented = virdi_augmented.loc[virdi_augmented.sold_month_and_year != "Jan_2018"]

virdi_augmented.loc[virdi_augmented.unit_type == "other", "unit_type"] = "apartment"

virdi_augmented = virdi_augmented.assign(title_lower=virdi_augmented.title.str.lower())

virdi_augmented = virdi_augmented.assign(needs_refurbishment=0)
virdi_augmented.needs_refurbishment = virdi_augmented.title_lower.str.contains("oppussingsobjekt")
virdi_augmented.needs_refurbishment = virdi_augmented.needs_refurbishment | virdi_augmented.title_lower.str.contains(
    "oppgraderingsbehov")
virdi_augmented.needs_refurbishment = virdi_augmented.needs_refurbishment | virdi_augmented.title_lower.str.contains(
    "oppussingsbehov")
virdi_augmented.needs_refurbishment = virdi_augmented.needs_refurbishment | virdi_augmented.title_lower.str.contains(
    "modernisering")
virdi_augmented.needs_refurbishment = virdi_augmented.needs_refurbishment.astype(int)

# ----------- CREATE NEW COLUMN year_group -----------#

year_labels = ['1820 - 1989', '1990 - 2004', '2005 - 2014', '2015 - 2020']

year_thresholds = [1820, 1990, 2005, 2015, 2020]

virdi_augmented['year_group'] = pd.cut(virdi_augmented.build_year, year_thresholds, right=False, labels=year_labels)

virdi_augmented = virdi_augmented.loc[~ pd.isnull(virdi_augmented.year_group)]
# ------------------ COMMON COST ----------------------#

virdi_augmented = virdi_augmented.assign(common_cost_per_m2=virdi_augmented.common_cost / virdi_augmented.prom)

virdi_augmented = virdi_augmented.assign(common_cost_is_high=np.where(virdi_augmented.common_cost > 4713, 1, 0))

# ------------------ HAS TWO OR THREE BEDROOMS -----------------#
virdi_augmented = virdi_augmented.assign(
    has_two_bedrooms=np.where((virdi_augmented.number_of_bedrooms == 2) & (virdi_augmented.prom < 60), 1, 0))
virdi_augmented = virdi_augmented.assign(
    has_three_bedrooms=np.where((virdi_augmented.number_of_bedrooms == 3) & (virdi_augmented.prom < 85), 1, 0))

# ------------------------------------------------------

virdi_augmented = virdi_augmented.assign(is_borettslag=~pd.isnull(virdi_augmented.borettslagetsnavn))
virdi_augmented.is_borettslag = virdi_augmented.is_borettslag.astype(int)

virdi_augmented = virdi_augmented.assign(has_garden=virdi_augmented.title_lower.str.contains("hage"))
virdi_augmented.has_garden = virdi_augmented.has_garden.astype(int)
virdi_augmented = virdi_augmented.assign(is_penthouse=virdi_augmented.title_lower.str.contains("toppleilighet"))
virdi_augmented.is_penthouse = virdi_augmented.is_penthouse.astype(int)
"""
virdi_augmented = virdi_augmented.assign(has_garage = virdi_augmented.title_lower.str.contains("garasje"))
virdi_augmented.has_garage = virdi_augmented.has_garage.astype(int)
virdi_augmented = virdi_augmented.assign(has_balcony = virdi_augmented.title_lower.str.contains("balkong"))
virdi_augmented.has_balcony = virdi_augmented.has_balcony.astype(int)
virdi_augmented = virdi_augmented.assign(has_fireplace = virdi_augmented.title_lower.str.contains("peis"))
virdi_augmented.has_fireplace = virdi_augmented.has_fireplace.astype(int)
"""
virdi_augmented = virdi_augmented.assign(has_terrace=virdi_augmented.title_lower.str.contains("terrasse"))
virdi_augmented.has_terrace = virdi_augmented.has_terrace.astype(int)

virdi_augmented = virdi_augmented.sample(frac=1)  # shuffle the dataset to do random training and test partition
virdi_augmented = virdi_augmented.reset_index(drop=True)

threshold = int(TRAINING_SET_SIZE * len(virdi_augmented))

# ------------
# Calculate repeat sales estimates
# ------------

virdi_augmented = virdi_augmented.assign(real_sold_date=virdi_augmented.real_sold_date.map(lambda x: pd.to_datetime(x)))
virdi_augmented = virdi_augmented.assign(
    prev_sale_date_1=virdi_augmented.prev_sale_date_1.map(lambda x: pd.to_datetime(x)))
virdi_augmented = virdi_augmented.assign(
    prev_sale_date_2=virdi_augmented.prev_sale_date_2.map(lambda x: pd.to_datetime(x)))
virdi_augmented = virdi_augmented.assign(
    prev_sale_date_3=virdi_augmented.prev_sale_date_3.map(lambda x: pd.to_datetime(x)))

# shift all prev sale dates because data is based on date of public registration
virdi_augmented = virdi_augmented.assign(prev_sale_date_1=np.where((~virdi_augmented.prev_sale_date_1.isnull()), \
                                                                   virdi_augmented.prev_sale_date_1 - pd.Timedelta(
                                                                       days=SOLD_TO_OFFICIAL_DIFF), \
                                                                   virdi_augmented.prev_sale_date_1))

virdi_augmented = virdi_augmented.assign(prev_sale_date_2=np.where((~virdi_augmented.prev_sale_date_2.isnull()), \
                                                                   virdi_augmented.prev_sale_date_2 - pd.Timedelta(
                                                                       days=SOLD_TO_OFFICIAL_DIFF), \
                                                                   virdi_augmented.prev_sale_date_2))

virdi_augmented = virdi_augmented.assign(prev_sale_date_3=np.where((~virdi_augmented.prev_sale_date_3.isnull()), \
                                                                   virdi_augmented.prev_sale_date_3 - pd.Timedelta(
                                                                       days=SOLD_TO_OFFICIAL_DIFF), \
                                                                   virdi_augmented.prev_sale_date_3))

virdi_augmented = virdi_augmented.assign(real_sold_quarter=virdi_augmented.real_sold_date.map(lambda x: get_quarter(x)))
virdi_augmented = virdi_augmented.assign(
    prev_sale_quarter_1=virdi_augmented.prev_sale_date_1.map(lambda x: get_quarter(x)))
virdi_augmented = virdi_augmented.assign(
    prev_sale_quarter_2=virdi_augmented.prev_sale_date_2.map(lambda x: get_quarter(x)))
virdi_augmented = virdi_augmented.assign(
    prev_sale_quarter_3=virdi_augmented.prev_sale_date_3.map(lambda x: get_quarter(x)))

print ""
print "Caluclating repeat sales estimates"

price_index_ssb = pd.read_csv("C:/Users/" + USER_STRING + "/data/price_index_oslo_ssb.csv", sep=";")

virdi_augmented = virdi_augmented.assign(real_sold_index=virdi_augmented.real_sold_quarter.map(
    lambda x: price_index_ssb.loc[price_index_ssb.quarter == x, "index"].iloc[0]))
print "Repeat 1"
virdi_augmented = virdi_augmented.assign(prev_sale_index_1=virdi_augmented.prev_sale_quarter_1.map(
    lambda x: price_index_ssb.loc[price_index_ssb.quarter == x, "index"].iloc[0], na_action='ignore'))
print "Repeat 2"
virdi_augmented = virdi_augmented.assign(prev_sale_index_2=virdi_augmented.prev_sale_quarter_2.map(
    lambda x: price_index_ssb.loc[price_index_ssb.quarter == x, "index"].iloc[0], na_action='ignore'))
print "Repeat 3"
virdi_augmented = virdi_augmented.assign(prev_sale_index_3=virdi_augmented.prev_sale_quarter_3.map(
    lambda x: price_index_ssb.loc[price_index_ssb.quarter == x, "index"].iloc[0], na_action='ignore'))

virdi_augmented = virdi_augmented.assign(prev_sale_estimate_1=(
                                                                          virdi_augmented.prev_sale_price_1 * virdi_augmented.real_sold_index / virdi_augmented.prev_sale_index_1) + virdi_augmented.common_debt)
virdi_augmented = virdi_augmented.assign(prev_sale_estimate_2=(
                                                                          virdi_augmented.prev_sale_price_2 * virdi_augmented.real_sold_index / virdi_augmented.prev_sale_index_2) + virdi_augmented.common_debt)
virdi_augmented = virdi_augmented.assign(prev_sale_estimate_3=(
                                                                          virdi_augmented.prev_sale_price_3 * virdi_augmented.real_sold_index / virdi_augmented.prev_sale_index_3) + virdi_augmented.common_debt)

virdi_augmented = virdi_augmented.assign(
    prev_sale_estimate_1=virdi_augmented.prev_sale_estimate_1.map(lambda x: int(x) if not np.isnan(x) else x))
virdi_augmented = virdi_augmented.assign(
    prev_sale_estimate_2=virdi_augmented.prev_sale_estimate_2.map(lambda x: int(x) if not np.isnan(x) else x))
virdi_augmented = virdi_augmented.assign(
    prev_sale_estimate_3=virdi_augmented.prev_sale_estimate_3.map(lambda x: int(x) if not np.isnan(x) else x))

print "Done calculating repeat sales estimates"

print "Data set shape before regression:"
print virdi_augmented.shape

alva_io.write_to_csv(virdi_augmented, "C:/Users/" + USER_STRING + "/data/virdi_aug_title_prev_sale_estimates.csv")

reg_start = time.time()

training = virdi_augmented[:threshold]
test = virdi_augmented[threshold:]

district_string_train = 'bydel_code,Treatment(reference="bfr")'
district_string_test = 'bydel_code,Treatment(reference="bfr")'

if kmeans_bool:
    print "Running K-means to construct new districts:"
    training, PRICE_FACTOR = kmeans.add_kmeans_districts(training, KMEANS_K)

    print "Predicting districts on test set using K-NN. K = " + str(KNN_K) + "."
    test = kmeans.predict_kmeans_districts(test, training, KNN_K)

    # kmeans_district_map.plot_districts(training, "bydel_code", 0)
    kmeans_district_map.plot_districts(training, "kmeans_cluster", PRICE_FACTOR)

    district_string_train = "kmeans_cluster"
    district_string_test = "kmeans_cluster_prediction"

wp_string = ""

if wp_bool:
    wp_string = " + WP"

# ------------
# Begin regressing
# ------------

print ""
print "Running regression"

reg_start = time.time()

y, X = dmatrices('log_price_plus_comdebt ~ C(size_group,Treatment(reference="40 -- 49")) + \
C(sold_month_and_year,Treatment(reference="Apr_2017")) + C(' + district_string_train + ') + \
C(unit_type,Treatment(reference="house")) + needs_refurbishment + year_group + has_garden + \
is_borettslag + is_penthouse + has_terrace + common_cost_is_high + has_two_bedrooms + \
has_three_bedrooms' + wp_string, data=training, return_type="dataframe")

# describe model
"""
print("y")
print(y.head(20))
print("x")
print(X.head(20))
print("---")
"""

if RUN_LAD:
    # LAD
    mod = sm.QuantReg(y, X, missing='drop')
    res = mod.fit(q=0.5)
else:
    # OLS
    mod = sm.OLS(y, X, missing='drop')
    res = mod.fit()

# magic
print res.summary()

"""
### ROBUST STANDARD ERROR
print ""
print("Robust standard error")
print(res.HC0_se)
"""

y_test, X_test = dmatrices('log_price_plus_comdebt ~ C(size_group,Treatment(reference="40 -- 49")) + \
C(sold_month_and_year,Treatment(reference="Apr_2017")) + C(' + district_string_test + ') + \
C(unit_type,Treatment(reference="house")) + needs_refurbishment + year_group + has_garden + \
is_borettslag + is_penthouse + has_terrace + common_cost_is_high + has_two_bedrooms + \
has_three_bedrooms' + wp_string, data=test, return_type="dataframe")

reg_predictions = res.predict(X_test)

print "Regression done"
print str(round(time.time() - reg_start, 4)) + "seconds elapsed"
# ------------
# Regression done
# ------------

training = training.assign(fitted_values=res.fittedvalues)
training = training.assign(fitted_values_nat=(np.exp(res.fittedvalues) * training.prom).astype(int))
training = training.assign(regression_residuals=res.resid)

test = test.assign(reg_prediction=reg_predictions)  # add regression prediction (y_reg) to test set
test = test.assign(reg_prediction_nat=(np.exp(reg_predictions) * test.prom).astype(int))
test = test.assign(regression_residual=test.log_price_plus_comdebt - reg_predictions)

alva_io.write_to_csv(training, "C:/Users/" + USER_STRING + "/data/training" + str(RUN_NUMBER) + ".csv")
alva_io.write_to_csv(test, "C:/Users/" + USER_STRING + "/data/test" + str(RUN_NUMBER) + ".csv")

# NOT NEEDED ANYMORE: Repeated for training set. Needed before to construct basic estimate and retrieve residual to use in comparable.
"""
### Remove repeated estimates if it deviates too much from fitted value (training set)

prev_sale_price_1_reasonable = (abs(training.prev_sale_estimate_1 - training.fitted_values_nat) / training.fitted_values_nat) < 0.25 ### using method from previous model
prev_sale_price_2_reasonable = (abs(training.prev_sale_estimate_2 - training.fitted_values_nat) / training.fitted_values_nat) < 0.25 ### using method from previous model
prev_sale_price_3_reasonable = (abs(training.prev_sale_estimate_3 - training.fitted_values_nat) / training.fitted_values_nat) < 0.25 ### using method from previous model

training = training.assign(prev_sale_estimate_3 = np.where(prev_sale_price_3_reasonable, training.prev_sale_estimate_3,np.nan))                             # set 3 to nan if invalid
training = training.assign(prev_sale_estimate_2 = np.where(prev_sale_price_2_reasonable, training.prev_sale_estimate_2,training.prev_sale_estimate_3))      # set 2 to 3 if 2 invalid
training = training.assign(prev_sale_estimate_3 = np.where(prev_sale_price_2_reasonable, training.prev_sale_estimate_3,np.nan))                             # remove 3 if 2 invalid
training = training.assign(prev_sale_estimate_1 = np.where(prev_sale_price_1_reasonable, training.prev_sale_estimate_1,training.prev_sale_estimate_2))      # set 1 to 2 if 1 invalid
training = training.assign(prev_sale_estimate_2 = np.where(prev_sale_price_1_reasonable, training.prev_sale_estimate_2,np.nan))                             # remove 2 if 1 invalid

prev_sale_price_2_reasonable = (abs(training.prev_sale_estimate_2 - training.fitted_values_nat) / training.fitted_values_nat) < 0.25 ### using method from previous model
training = training.assign(prev_sale_estimate_2 = np.where(prev_sale_price_2_reasonable, training.prev_sale_estimate_2,training.prev_sale_estimate_3))      # set 2 to 3 if 2 invalid
training = training.assign(prev_sale_estimate_3 = np.where(prev_sale_price_2_reasonable, training.prev_sale_estimate_3,np.nan))                             # remove 3 if 2 invalid

prev_sale_price_1_exists = ~pd.isnull(training.prev_sale_estimate_1)
prev_sale_price_2_exists = ~pd.isnull(training.prev_sale_estimate_2)
prev_sale_price_3_exists = ~pd.isnull(training.prev_sale_estimate_3)

number_of_prev_sales = prev_sale_price_1_exists.astype(int) + prev_sale_price_2_exists.astype(int) + prev_sale_price_3_exists.astype(int)

training = training.assign(number_of_prev_sales = number_of_prev_sales)
"""

### Remove repeated estimates if it deviates too much from fitted value (test set)
### Count number of previous sales, test set

prev_sale_price_1_reasonable = (abs(
    test.prev_sale_estimate_1 - test.reg_prediction_nat) / test.reg_prediction_nat) < 0.25  ### using method from previous model
prev_sale_price_2_reasonable = (abs(
    test.prev_sale_estimate_2 - test.reg_prediction_nat) / test.reg_prediction_nat) < 0.25  ### using method from previous model
prev_sale_price_3_reasonable = (abs(
    test.prev_sale_estimate_3 - test.reg_prediction_nat) / test.reg_prediction_nat) < 0.25  ### using method from previous model

test = test.assign(prev_sale_estimate_3=np.where(prev_sale_price_3_reasonable, test.prev_sale_estimate_3,
                                                 np.nan))  # set 3 to nan if invalid
test = test.assign(prev_sale_estimate_2=np.where(prev_sale_price_2_reasonable, test.prev_sale_estimate_2,
                                                 test.prev_sale_estimate_3))  # set 2 to 3 if 2 invalid
test = test.assign(prev_sale_estimate_3=np.where(prev_sale_price_2_reasonable, test.prev_sale_estimate_3,
                                                 np.nan))  # remove 3 if 2 invalid
test = test.assign(prev_sale_estimate_1=np.where(prev_sale_price_1_reasonable, test.prev_sale_estimate_1,
                                                 test.prev_sale_estimate_2))  # set 1 to 2 if 1 invalid
test = test.assign(prev_sale_estimate_2=np.where(prev_sale_price_1_reasonable, test.prev_sale_estimate_2,
                                                 np.nan))  # remove 2 if 1 invalid

prev_sale_price_2_reasonable = (abs(
    test.prev_sale_estimate_2 - test.reg_prediction_nat) / test.reg_prediction_nat) < 0.25  ### using method from previous model
test = test.assign(prev_sale_estimate_2=np.where(prev_sale_price_2_reasonable, test.prev_sale_estimate_2,
                                                 test.prev_sale_estimate_3))  # set 2 to 3 if 2 invalid
test = test.assign(prev_sale_estimate_3=np.where(prev_sale_price_2_reasonable, test.prev_sale_estimate_3,
                                                 np.nan))  # remove 3 if 2 invalid

prev_sale_price_1_exists = ~pd.isnull(test.prev_sale_estimate_1)
prev_sale_price_2_exists = ~pd.isnull(test.prev_sale_estimate_2)
prev_sale_price_3_exists = ~pd.isnull(test.prev_sale_estimate_3)

number_of_prev_sales = prev_sale_price_1_exists.astype(int) + prev_sale_price_2_exists.astype(
    int) + prev_sale_price_3_exists.astype(int)

test = test.assign(number_of_prev_sales=number_of_prev_sales)

test = test.reset_index()
training = training.reset_index()

# ------------
# Construct basic estimates for test set
# ------------

#print test.sort_values(by = ["regression_residual"], ascending=True).head(20)


### Remove repeated estimates if it deviates too much from fitted value (test set)

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



score = pd.DataFrame()

score = score.assign(true_value=test["Total price"])

score = score.assign(reg_prediction=test.reg_prediction_nat)
score = score.assign(reg_deviation=score.true_value - score.reg_prediction)
score = score.assign(reg_deviation_percentage=abs(score.reg_deviation / score.true_value))
score = score.assign(rep_1_prediction=test.prev_sale_estimate_1)
score = score.assign(rep_2_prediction=test.prev_sale_estimate_2)
score = score.assign(rep_3_prediction=test.prev_sale_estimate_3)
score = score.assign(rep_1_deviation=score.true_value - score.rep_1_prediction)
score = score.assign(rep_2_deviation=score.true_value - score.rep_2_prediction)
score = score.assign(rep_3_deviation=score.true_value - score.rep_3_prediction)
score = score.assign(rep_1_deviation_percentage=abs(score.rep_1_deviation / score.true_value))
score = score.assign(rep_2_deviation_percentage=abs(score.rep_2_deviation / score.true_value))
score = score.assign(rep_3_deviation_percentage=abs(score.rep_3_deviation / score.true_value))

score = score.assign(basic_prediction = test.basic_estimate)
score = score.assign(basic_deviation = score.true_value - score.basic_prediction)
score = score.assign(basic_deviation_percentage = abs(score.basic_deviation / score.true_value))


regression_type = "Regression"
if wp_bool:
    regression_type = "Autoregressive"

csv_results = pd.DataFrame(columns = ["Ordinary" + regression_type, "Repeat Combination", "Time"])
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
    moran, geary = spatial_measures.get_i_and_c(test, 100, "regression_residual")
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
    moran, geary = spatial_measures.get_i_and_c(test, 100, "basic_estimate_residual")
    # basic_estimate_residual
    print ""
    print "Moran's I: " + str(100 * round(moran, 4)) + "% and Geary's C: " + str(100 * round(geary, 4)) + "%."

repeat_results = repeat_results.append(pd.Series([moran, geary]), ignore_index=True)

csv_results["Ordinary" + regression_type] = ordinary_results
csv_results["Repeat Combination"] = repeat_results
csv_results["Time"]  = pd.Series(datetime.datetime.now().strftime ("%c"))

csv_file_name = "C:/Users/" + USER_STRING + "/data/" + regression_type

if kmeans_bool:
    print "Number of k-means districts was: " + str(KMEANS_K)
    print "K-means price factor: " + str(PRICE_FACTOR)
    csv_file_name += "_kmeans_results.csv"
else:
    print "Used administrative districts"
    csv_file_name += "_admin_results.csv"
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
