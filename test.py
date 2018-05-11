# -*- coding: utf-8 -*-

from scipy.spatial.distance import pdist, squareform
import time
import kmeans
import alva_io, address
import pandas as pd
import numpy as np
import calendar
from patsy import dmatrices
import moran
import statsmodels.api as sm
import matplotlib.pyplot as plt


SOLD_TO_OFFICIAL_DIFF = 55
TRAINING_SET_SIZE = 0.8
COMPARABLE_SET_SIZE = 10
RUN_LAD = True
REG_SHARE = 0.5
KMEANS_K = 15
KNN_K = 5

MORANS_SET_SIZE = 10

def get_quarter(date):
    if pd.isnull(date):
        return np.nan
    year = date.year
    month = date.month
    quarter = 1 + ((month - 1) // 3)
    quarter_string = str(year) + "_Q" + str(quarter)
    return quarter_string




if RUN_LAD:
    print "Running LAD with basic model regression share: " + str(REG_SHARE * 100) + "%."
else:
    print "Running OLS with basic model regression share: " + str(REG_SHARE * 100) + "%."


virdi_augmented = pd.read_csv("C:/Users/tobiasrp/data/virdi_aug_title_prev_sale.csv")
virdi_augmented = virdi_augmented.assign(real_sold_date = pd.to_datetime(virdi_augmented["real_sold_date"],format="%Y-%m-%d"))


# virdi = virdi.loc[virdi["kr/m2"] > ] # cutting on sqm price

virdi_augmented = virdi_augmented.assign(log_price_plus_comdebt = np.log(virdi_augmented["kr/m2"]))

# ----- CREATE NEW COLUMN size_group ------ #

size_group_size = 10 #change this only to change group sizes
size_labels = ["10 -- 29"]
size_labels += [ "{0} -- {1}".format(i, i + size_group_size - 1) for i in range(30, 150, size_group_size) ]
size_labels.append("150 -- 179")
size_labels.append("180 -- ")

size_thresholds = [10]
size_thresholds += [i for i in range(30,151,size_group_size)] + [180] + [500]

virdi_augmented['size_group'] = pd.cut(virdi_augmented.prom, size_thresholds, right=False, labels=size_labels)


#----- CREATE NEW COLUMN sold_month ------#

#add sold_month column for discovery of seasonal effects

virdi_augmented = virdi_augmented.assign(sold_month = virdi_augmented.real_sold_date.map(lambda x: x.month))
virdi_augmented = virdi_augmented.assign(sold_year = virdi_augmented.real_sold_date.map(lambda x: x.year))
virdi_augmented = virdi_augmented.assign(sold_month = virdi_augmented.sold_month.apply(lambda x: calendar.month_abbr[x])) #text instead of int
virdi_augmented = virdi_augmented.assign(sold_month_and_year = virdi_augmented.sold_month + "_" + virdi_augmented.sold_year.map(str))

#delete sold_date and official_date now that real_sold_date covers both
del virdi_augmented["sold_date"]
del virdi_augmented["official_date"]
del virdi_augmented["register_date"]

# ------------------
#   DO REGRESSION
# ------------------

# virdi_augmented = virdi_augmented[["log_price_plus_comdebt","size_group","sold_month_and_year","bydel_code","unit_type", "prom","Total price","coord_x","coord_y"]]

columns_to_count = ["size_group","sold_month_and_year","bydel_code","unit_type"]

# virdi_augmented = virdi_augmented.loc[virdi_augmented.bydel_code != "SEN"]
virdi_augmented.loc[virdi_augmented.bydel_code == 'SEN', 'bydel_code'] = "bsh" # reassign SENTRUM to St. Hanshaugen

virdi_augmented = virdi_augmented.loc[virdi_augmented.bydel_code != "MAR"]

virdi_augmented = virdi_augmented.loc[virdi_augmented.sold_month_and_year != "Feb_2018"]
virdi_augmented = virdi_augmented.loc[virdi_augmented.sold_month_and_year != "Jan_2018"]

virdi_augmented.loc[virdi_augmented.unit_type == "other", "unit_type"] = "apartment"
#virdi_augmented = virdi_augmented.loc[virdi_augmented.unit_type != "other"]

virdi_augmented = virdi_augmented.assign(title_lower = virdi_augmented.title.str.lower())

virdi_augmented = virdi_augmented.assign(needs_refurbishment = 0)
virdi_augmented.needs_refurbishment = virdi_augmented.title_lower.str.contains("oppussingsobjekt")
virdi_augmented.needs_refurbishment = virdi_augmented.needs_refurbishment | virdi_augmented.title_lower.str.contains("oppgraderingsbehov")
virdi_augmented.needs_refurbishment = virdi_augmented.needs_refurbishment | virdi_augmented.title_lower.str.contains("oppussingsbehov")
virdi_augmented.needs_refurbishment = virdi_augmented.needs_refurbishment.astype(int)

#----------- CREATE NEW COLUMN year_group -----------#

year_labels = ['1820 - 1989', '1990 - 2004', '2005 - 2014', '2015 - 2020']

year_thresholds = [1820,1990,2005,2015,2020]

virdi_augmented['year_group'] = pd.cut(virdi_augmented.build_year, year_thresholds, right=False, labels=year_labels)
virdi_augmented = virdi_augmented.loc[~ pd.isnull(virdi_augmented.year_group)]

#------------------ COMMON COST ----------------------#

virdi_augmented = virdi_augmented.assign(common_cost_per_m2 = virdi_augmented.common_cost / virdi_augmented.prom)

virdi_augmented = virdi_augmented.assign(common_cost_is_high = np.where(virdi_augmented.common_cost > 4713, 1, 0))

#------------------ HAS TWO OR THREE BEDROOMS -----------------#
virdi_augmented = virdi_augmented.assign(has_two_bedrooms = np.where((virdi_augmented.number_of_bedrooms == 2) & (virdi_augmented.prom < 60),1,0))
virdi_augmented = virdi_augmented.assign(has_three_bedrooms = np.where((virdi_augmented.number_of_bedrooms == 3) & (virdi_augmented.prom < 85),1,0))

#------------------------------------------------------

virdi_augmented = virdi_augmented.assign(is_borettslag = ~pd.isnull(virdi_augmented.borettslagetsnavn))
virdi_augmented.is_borettslag = virdi_augmented.is_borettslag.astype(int)



virdi_augmented = virdi_augmented.assign(has_garden = virdi_augmented.title_lower.str.contains("hage"))
virdi_augmented.has_garden = virdi_augmented.has_garden.astype(int)
virdi_augmented = virdi_augmented.assign(is_penthouse = virdi_augmented.title_lower.str.contains("toppleilighet"))
virdi_augmented.is_penthouse = virdi_augmented.is_penthouse.astype(int)
"""
virdi_augmented = virdi_augmented.assign(has_garage = virdi_augmented.title_lower.str.contains("garasje"))
virdi_augmented.has_garage = virdi_augmented.has_garage.astype(int)
virdi_augmented = virdi_augmented.assign(has_balcony = virdi_augmented.title_lower.str.contains("balkong"))
virdi_augmented.has_balcony = virdi_augmented.has_balcony.astype(int)
virdi_augmented = virdi_augmented.assign(has_fireplace = virdi_augmented.title_lower.str.contains("peis"))
virdi_augmented.has_fireplace = virdi_augmented.has_fireplace.astype(int)
"""
virdi_augmented = virdi_augmented.assign(has_terrace = virdi_augmented.title_lower.str.contains("terrasse"))
virdi_augmented.has_terrace = virdi_augmented.has_terrace.astype(int)





reg_start = time.time()

virdi_augmented = virdi_augmented.sample(frac = 1)

threshold = int(TRAINING_SET_SIZE * len(virdi_augmented))
training = virdi_augmented[:threshold]
test = virdi_augmented[threshold:]

"""
print "Running K-means to construct new districts:"
training = kmeans.add_kmeans_districts(training, KMEANS_K)

print "Predicting districts on test set using K-NN. K = " + str(KNN_K) + "."
test = kmeans.predict_kmeans_districts(test, training, KNN_K)
"""

test = test.reset_index(drop=True)

y,X = dmatrices('log_price_plus_comdebt ~ C(size_group,Treatment(reference="40 -- 49")) + \
C(sold_month_and_year,Treatment(reference="Apr_2017")) + \
C(unit_type,Treatment(reference="house")) + needs_refurbishment + year_group + has_garden + \
is_borettslag + is_penthouse + has_terrace + common_cost_is_high + has_two_bedrooms + \
has_three_bedrooms', data=training, return_type = "dataframe")


#describe model
"""
print("y")
print(y.head(20))
print("x")
print(X.head(20))
print("---")
"""

if RUN_LAD:
    #LAD
    mod = sm.QuantReg(y,X, missing='drop')
    res = mod.fit(q=0.5)
else:
    #OLS
    mod = sm.OLS(y,X, missing='drop')
    res = mod.fit()

#magic
#print(res.summary())

"""
### ROBUST STANDARD ERROR
print("-----------------")
print("Robust standard error")
print()
print(res.HC0_se)
"""

y_test,X_test = dmatrices('log_price_plus_comdebt ~ C(size_group,Treatment(reference="40 -- 49")) + \
C(sold_month_and_year,Treatment(reference="Apr_2017")) + \
C(unit_type,Treatment(reference="house")) + needs_refurbishment + year_group + has_garden + \
is_borettslag + is_penthouse + has_terrace + common_cost_is_high + has_two_bedrooms + \
has_three_bedrooms', data=test, return_type = "dataframe")

reg_predictions = res.predict(X_test)
test = test.assign(reg_resids = test.log_price_plus_comdebt - reg_predictions)

print "Regression done"
print str(round(time.time() - reg_start,4)) + "seconds elapsed"
print
print ""


moran, geary = moran.i(test, 100, "reg_resids")
print "MORAN'S I: " + str(round(moran,4))
print "GEARY'S C: " + str(round(geary,4))