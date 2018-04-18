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

SOLD_TO_OFFICIAL_DIFF = 55
TRAINING_SET_SIZE = 0.8
COMPARABLE_SET_SIZE = 20

def get_quarter(date):
    if pd.isnull(date):
        return np.nan
    year = date.year
    month = date.month
    quarter = 1 + ((month - 1) // 3)
    quarter_string = str(year) + "_Q" + str(quarter)
    return quarter_string


if raw_input("Enter to run without new initialization") != "":

    virdi_df = alva_io.get_dataframe_from_excel("C:/Users/tobiasrp/data/20180220 Transactions Virdi v2.xlsx")

    address_df = address.get_address()

    virdi_augmented = pd.merge(virdi_df, address_df, on="id")

    print("Dropping duplicates")
    virdi_augmented = virdi_augmented.drop_duplicates(subset="ad_code", keep=False)

    print("Dropping NaN in ad_code")
    print "print1", virdi_augmented.shape
    virdi_augmented = virdi_augmented.dropna(subset = ["ad_code"])
    print "print2", virdi_augmented.shape

    pd.DataFrame.to_csv(virdi_augmented, "C:/Users/tobiasrp/data/virdi_augmented.csv")
    #after these operations, scrape FINN for additional data to create virdi augmented with title

else:
    virdi_augmented = pd.read_csv("C:/Users/tobiasrp/data/virdi_augmented_with_title.csv",index_col=0)

if raw_input("Enter to run without mapping previous sales") != "":

    #----------- CREATE NEW COLUMN real_sold_date -----------#

    #makes sold_date on datetime format
    virdi_augmented = virdi_augmented.assign(sold_date = pd.to_datetime(virdi_augmented["sold_date"],format="%Y-%m-%d"))

    #makes off._date on datetime format
    virdi_augmented = virdi_augmented.assign(official_date = pd.to_datetime(virdi_augmented["official_date"],format="%Y-%m-%d"))

    #sets new col real_sold_date equal to minimum of sold_date and official_date (with offset of x days)
    virdi_augmented = virdi_augmented.assign(real_sold_date = np.where((virdi_augmented["sold_date"] > virdi_augmented["official_date"]), \
                                                                       virdi_augmented["official_date"] - pd.Timedelta(days = SOLD_TO_OFFICIAL_DIFF), \
                                                                       virdi_augmented["sold_date"]))


    #sets real_sold_date equal to official_date if sold_date is NaT (not a problem in the opposite case for some reason)
    virdi_augmented = virdi_augmented.assign(real_sold_date = np.where((virdi_augmented["real_sold_date"].isnull()), \
                                                                       virdi_augmented["official_date"]- pd.Timedelta(days = SOLD_TO_OFFICIAL_DIFF), \
                                                                       virdi_augmented["real_sold_date"]))

    ### ADD PREVIOUS SALES

    prev_sales = address.get_previous_sales()
    prev_sales = prev_sales[["id", "official_date", "official_price"]]

    prev_sales = prev_sales.rename(columns={'official_price': 'prev_sale_price_1'})
    prev_sales = prev_sales.rename(columns={'official_date': 'prev_sale_date_1'})

    virdi_augmented = pd.merge(virdi_augmented, prev_sales, how='left', on='id')

    print "-------------"

    virdi_augmented = virdi_augmented.assign(prev_sale_price_2=np.nan)
    virdi_augmented = virdi_augmented.assign(prev_sale_date_2=np.nan)
    virdi_augmented = virdi_augmented.assign(prev_sale_price_3=np.nan)
    virdi_augmented = virdi_augmented.assign(prev_sale_date_3=np.nan)

    print virdi_augmented[["id", "real_sold_date", "Finalprice", "prev_sale_date_1", "prev_sale_price_1"]].head(50)
    virdi_augmented.loc[virdi_augmented.Finalprice == virdi_augmented.prev_sale_price_1, 'prev_sale_date_1'] = np.nan
    virdi_augmented.loc[virdi_augmented.Finalprice == virdi_augmented.prev_sale_price_1, 'prev_sale_price_1'] = np.nan

    virdi_augmented.loc[virdi_augmented.real_sold_date < virdi_augmented.prev_sale_date_1, 'prev_sale_price_1'] = np.nan
    virdi_augmented.loc[virdi_augmented.real_sold_date < virdi_augmented.prev_sale_date_1, 'prev_sale_date_1'] = np.nan

    print virdi_augmented[["id", "real_sold_date", "Finalprice", "prev_sale_date_1", "prev_sale_price_1"]].head(50)

    #########################
    ## MAP PREVIOUS SALES ##
    #########################

    count = 0
    count_adjustment = 0
    delete_rows = []
    progress_count = 0.0
    size = float(len(virdi_augmented.index))
    old_progress, new_progress = 0, 0

    print "Mapping previous sales for " + str(size) + " apartments."
    print "0%",
    for index, row in virdi_augmented.iterrows():
        id = row.id
        prev_id = virdi_augmented.iloc[index - 1].id

        if id == prev_id and count < 3:
            count += 1
            prev_sale_price_col_name = "prev_sale_price_" + str(count - count_adjustment)
            prev_sale_date_col_name = "prev_sale_date_" + str(count - count_adjustment)
            if pd.isnull(row.prev_sale_price_1):
                #print "null: " + str(index)
                count_adjustment += 1
            else:
                virdi_augmented[prev_sale_price_col_name][index - count] = row.prev_sale_price_1
                virdi_augmented[prev_sale_date_col_name][index - count] = row.prev_sale_date_1
                count_adjustment = 0
            delete_rows.append(index)
        elif id != prev_id:
            if not pd.isnull(row.prev_sale_price_1):
                #print "not null: " + str(index)
                count_adjustment = -1
            else:
                count_adjustment = 0
            count = 0
        else:
            delete_rows.append(index)
            # count += 1 not necessary

        new_progress = round(progress_count / size, 2)
        if old_progress != new_progress:
            if (int(100 * new_progress)) % 10 == 0:
                print str(int(100 * new_progress)) + "%",
            else:
                print "|",
        old_progress = new_progress

        progress_count += 1

    virdi_augmented = virdi_augmented.drop(virdi_augmented.index[delete_rows])
    alva_io.write_to_csv(virdi_augmented,"C:/Users/tobiasrp/data/virdi_aug_title_prev_sale.csv")

    print ""
    print ""

    #########################

else:
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

"""
for c in columns_to_count:
    print(c)
    print(virdi_augmented[c].value_counts())
    print("-------------------")
"""
"""
count = 0.0
size = float(len(virdi_augmented.index))
old_progress, new_progress = 0,0
WP_list = []

print "Mapping distances"
print "0%",
for_loop_start = time.time()

for index, r in virdi_augmented.iterrows():

    comparable_matrix = virdi_augmented.loc[virdi_augmented.size_group == r.size_group]

    if r.unit_type == "apartment":
        comparable_matrix = comparable_matrix.loc[comparable_matrix.bydel_code == r.bydel_code]

    comparable_matrix = comparable_matrix.loc[comparable_matrix.real_sold_date < r.real_sold_date] # comparable sale must have occured earlier in time
    comparable_matrix = comparable_matrix.loc[comparable_matrix.real_sold_date + pd.Timedelta(days=90) > r.real_sold_date] # comparable sale must have occured during the last 90 days

    comparable_matrix = comparable_matrix.append(r)

    distance_matrix = squareform(pdist(comparable_matrix[["coord_x", "coord_y"]]))

    max_distance = 0.000025 # about 1.5 meters
    close_ids = []

    while len(close_ids) < COMPARABLE_SET_SIZE + 1 and max_distance <= 0.0025: # about 150 meters
        close_points = (distance_matrix[-1] < max_distance)
        close_points[-1] = False  # exclude itself - if present, always the last element
        close_ids_comparable_matrix = [comparable_matrix.iloc[i].name for i, close in enumerate(close_points) if close]
        close_ids = [i for i, close in enumerate(close_points) if close]
        max_distance *= 10

    close_ids_comparable_matrix = close_ids_comparable_matrix[:COMPARABLE_SET_SIZE]
    close_ids = close_ids[:COMPARABLE_SET_SIZE]

    p_comparables = comparable_matrix.loc[close_ids_comparable_matrix].log_price_plus_comdebt

    WP = p_comparables.mean()

    #-------

    distances = distance_matrix[-1][[close_ids]]
    if len(distances) > 0:
        if distances.max() == 0:
            distances[distances == 0] = 0.000025 # value is irrelevant
        else:
            distances[distances == 0] = distances[distances > 0].min()
    distances = 1 / distances

    p_times_distance = distances * comparable_matrix.loc[close_ids_comparable_matrix].log_price_plus_comdebt

    WP = p_times_distance.sum() / distances.sum()

    # -------

    WP_list.append(WP)

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
print "Done. " + str(round(time.time() - for_loop_start )) + " seconds elapsed."

WP_column = pd.Series(WP_list)

virdi_augmented = virdi_augmented.assign(WP = WP_column)

virdi_augmented = virdi_augmented.dropna(subset = ["WP"])

alva_io.write_to_csv(virdi_augmented,"C:/Users/tobiasrp/data/virdi_augmented_with_WP.csv")
"""

# ------------
# Calculate repeat sales estimates
# ------------

virdi_augmented = virdi_augmented.sample(frac=1) # shuffle the dataset to do random training and test partition

virdi_augmented = virdi_augmented.assign(real_sold_date = virdi_augmented.real_sold_date.map(lambda x: pd.to_datetime(x)))
virdi_augmented = virdi_augmented.assign(prev_sale_date_1 = virdi_augmented.prev_sale_date_1.map(lambda x: pd.to_datetime(x)))
virdi_augmented = virdi_augmented.assign(prev_sale_date_2 = virdi_augmented.prev_sale_date_2.map(lambda x: pd.to_datetime(x)))
virdi_augmented = virdi_augmented.assign(prev_sale_date_3 = virdi_augmented.prev_sale_date_3.map(lambda x: pd.to_datetime(x)))

# shift all prev sale dates because data is based on date of public registration
virdi_augmented = virdi_augmented.assign(prev_sale_date_1=np.where((~virdi_augmented.prev_sale_date_1.isnull()), \
                                                                   virdi_augmented.prev_sale_date_1 - pd.Timedelta(days=SOLD_TO_OFFICIAL_DIFF), \
                                                                   virdi_augmented.prev_sale_date_1))

virdi_augmented = virdi_augmented.assign(prev_sale_date_2=np.where((~virdi_augmented.prev_sale_date_2.isnull()), \
                                                                   virdi_augmented.prev_sale_date_2 - pd.Timedelta(days=SOLD_TO_OFFICIAL_DIFF), \
                                                                   virdi_augmented.prev_sale_date_2))

virdi_augmented = virdi_augmented.assign(prev_sale_date_3=np.where((~virdi_augmented.prev_sale_date_3.isnull()), \
                                                                   virdi_augmented.prev_sale_date_3 - pd.Timedelta(days=SOLD_TO_OFFICIAL_DIFF), \
                                                                   virdi_augmented.prev_sale_date_3))

virdi_augmented = virdi_augmented.assign(real_sold_quarter = virdi_augmented.real_sold_date.map(lambda x: get_quarter(x)))
virdi_augmented = virdi_augmented.assign(prev_sale_quarter_1 = virdi_augmented.prev_sale_date_1.map(lambda x: get_quarter(x)))
virdi_augmented = virdi_augmented.assign(prev_sale_quarter_2 = virdi_augmented.prev_sale_date_2.map(lambda x: get_quarter(x)))
virdi_augmented = virdi_augmented.assign(prev_sale_quarter_3 = virdi_augmented.prev_sale_date_3.map(lambda x: get_quarter(x)))

print ""
print "Caluclating repeat sales estimates"

price_index_ssb = pd.read_csv("C:/Users/tobiasrp/data/price_index_oslo_ssb.csv", sep=";")

virdi_augmented = virdi_augmented.assign(real_sold_index = virdi_augmented.real_sold_quarter.map(lambda x: price_index_ssb.loc[price_index_ssb.quarter == x, "index"].iloc[0]))
print "Repeat 1"
virdi_augmented = virdi_augmented.assign(prev_sale_index_1 = virdi_augmented.prev_sale_quarter_1.map(lambda x: price_index_ssb.loc[price_index_ssb.quarter == x, "index"].iloc[0], na_action = 'ignore'))
print "Repeat 2"
virdi_augmented = virdi_augmented.assign(prev_sale_index_2 = virdi_augmented.prev_sale_quarter_2.map(lambda x: price_index_ssb.loc[price_index_ssb.quarter == x, "index"].iloc[0], na_action = 'ignore'))
print "Repeat 3"
virdi_augmented = virdi_augmented.assign(prev_sale_index_3 = virdi_augmented.prev_sale_quarter_3.map(lambda x: price_index_ssb.loc[price_index_ssb.quarter == x, "index"].iloc[0], na_action = 'ignore'))

virdi_augmented = virdi_augmented.assign(prev_sale_estimate_1 = (virdi_augmented.prev_sale_price_1 * virdi_augmented.real_sold_index / virdi_augmented.prev_sale_index_1) + virdi_augmented.common_debt)
virdi_augmented = virdi_augmented.assign(prev_sale_estimate_2 = (virdi_augmented.prev_sale_price_2 * virdi_augmented.real_sold_index / virdi_augmented.prev_sale_index_2) + virdi_augmented.common_debt)
virdi_augmented = virdi_augmented.assign(prev_sale_estimate_3 = (virdi_augmented.prev_sale_price_3 * virdi_augmented.real_sold_index / virdi_augmented.prev_sale_index_3) + virdi_augmented.common_debt)

virdi_augmented = virdi_augmented.assign(prev_sale_estimate_1 = virdi_augmented.prev_sale_estimate_1.map(lambda x: int(x) if not np.isnan(x) else x))
virdi_augmented = virdi_augmented.assign(prev_sale_estimate_2 = virdi_augmented.prev_sale_estimate_2.map(lambda x: int(x) if not np.isnan(x) else x))
virdi_augmented = virdi_augmented.assign(prev_sale_estimate_3 = virdi_augmented.prev_sale_estimate_3.map(lambda x: int(x) if not np.isnan(x) else x))

print "Done calculating repeat sales estimates"
alva_io.write_to_csv(virdi_augmented,"C:/Users/tobiasrp/data/virdi_aug_title_prev_sale_estimates.csv")

print ""
print "Running regression"
# ------------
# Begin splitting and regressing
# ------------

"""
for c in virdi_augmented.columns:
    print "-------"
    print c
    print virdi_augmented[c].isnull().sum()
"""

threshold = int(TRAINING_SET_SIZE * len(virdi_augmented))
training = virdi_augmented[:threshold]
test = virdi_augmented[threshold:]

y,X = dmatrices('log_price_plus_comdebt ~ C(size_group,Treatment(reference="40 -- 49")) + \
C(sold_month_and_year,Treatment(reference="Apr_2017")) + C(bydel_code,Treatment(reference="bfr")) + \
C(unit_type,Treatment(reference="house")) + needs_refurbishment + year_group + has_garden + \
is_borettslag + is_penthouse + has_terrace + common_cost_is_high + has_two_bedrooms + \
has_three_bedrooms', data=training, return_type = "dataframe") # excluding build year until videre


#describe model
"""
print("y")
print(y.head(20))
print("x")
print(X.head(20))
print("---")
"""
mod = sm.OLS(y,X,missing = 'drop')

#fit model
res = mod.fit()

#magic
print(res.summary())

"""
### ROBUST STANDARD ERROR
print("-----------------")
print("Robust standard error")
print()
print(res.HC0_se)
"""

y_test,X_test = dmatrices('log_price_plus_comdebt ~ C(size_group,Treatment(reference="40 -- 49")) + \
C(sold_month_and_year,Treatment(reference="Apr_2017")) + C(bydel_code,Treatment(reference="bfr")) + \
C(unit_type,Treatment(reference="house")) + needs_refurbishment + year_group + has_garden + \
is_borettslag + is_penthouse + has_terrace + common_cost_is_high + has_two_bedrooms + \
has_three_bedrooms', data=test, return_type = "dataframe")

reg_predictions = res.predict(X_test)

print "Regression done"
print ""
# ------------
# Regression done
# ------------

# ------------
# Construct basic estimates for training set
# ------------

training = training.assign(regression_residuals = res.resid)

training = training.assign(fitted_values = res.fittedvalues)
fitted_training_values_natural = (np.exp(res.fittedvalues) * training.prom).astype(int)
training = training.assign(fitted_values_nat = fitted_training_values_natural)

print "Constructing basic estimates for training set"
### Remove repeated estimates if it deviates too much from fitted value (training set)

prev_sale_price_1_reasonable = (abs(training.prev_sale_estimate_1 - fitted_training_values_natural) / fitted_training_values_natural) < 0.25 ### using method from previous model
prev_sale_price_2_reasonable = (abs(training.prev_sale_estimate_2 - fitted_training_values_natural) / fitted_training_values_natural) < 0.25 ### using method from previous model
prev_sale_price_3_reasonable = (abs(training.prev_sale_estimate_3 - fitted_training_values_natural) / fitted_training_values_natural) < 0.25 ### using method from previous model

training = training.assign(prev_sale_estimate_3 = np.where(prev_sale_price_3_reasonable, training.prev_sale_estimate_3,np.nan))                             # set 3 to nan if invalid
training = training.assign(prev_sale_estimate_2 = np.where(prev_sale_price_2_reasonable, training.prev_sale_estimate_2,training.prev_sale_estimate_3))      # set 2 to 3 if 2 invalid
training = training.assign(prev_sale_estimate_3 = np.where(prev_sale_price_2_reasonable, training.prev_sale_estimate_3,np.nan))                             # remove 3 if 2 invalid
training = training.assign(prev_sale_estimate_1 = np.where(prev_sale_price_1_reasonable, training.prev_sale_estimate_1,training.prev_sale_estimate_2))      # set 1 to 2 if 1 invalid
training = training.assign(prev_sale_estimate_2 = np.where(prev_sale_price_1_reasonable, training.prev_sale_estimate_2,np.nan))                             # remove 2 if 1 invalid

prev_sale_price_2_reasonable = (abs(training.prev_sale_estimate_2 - fitted_training_values_natural) / fitted_training_values_natural) < 0.25 ### using method from previous model
training = training.assign(prev_sale_estimate_2 = np.where(prev_sale_price_2_reasonable, training.prev_sale_estimate_2,training.prev_sale_estimate_3))      # set 2 to 3 if 2 invalid
training = training.assign(prev_sale_estimate_3 = np.where(prev_sale_price_2_reasonable, training.prev_sale_estimate_3,np.nan))                             # remove 3 if 2 invalid

prev_sale_price_1_exists = ~pd.isnull(training.prev_sale_estimate_1)
prev_sale_price_2_exists = ~pd.isnull(training.prev_sale_estimate_2)
prev_sale_price_3_exists = ~pd.isnull(training.prev_sale_estimate_3)

number_of_prev_sales = prev_sale_price_1_exists.astype(int) + prev_sale_price_2_exists.astype(int) + prev_sale_price_3_exists.astype(int)

training = training.assign(number_of_prev_sales = number_of_prev_sales)

number_of_prev_sales_dummy = pd.get_dummies(training.number_of_prev_sales)

basic_est_0 = training.fitted_values_nat
basic_est_1 = 0.5 * training.fitted_values_nat + 0.5 * training.prev_sale_estimate_1
basic_est_2 = 0.5 * training.fitted_values_nat + 0.5 * (0.9 * training.prev_sale_estimate_1 + 0.1 * training.prev_sale_estimate_2)
basic_est_3 = 0.5 * training.fitted_values_nat + 0.5 * (0.85 * training.prev_sale_estimate_1 + 0.15 * (0.8 * training.prev_sale_estimate_2 + 0.2 * training.prev_sale_estimate_3))

basic_estimate_matrix = pd.concat([basic_est_0,basic_est_1,basic_est_2,basic_est_3], axis = 1)

basic_estimate_matrix = pd.DataFrame(data=np.where(number_of_prev_sales_dummy, basic_estimate_matrix, 0))
basic_estimate = basic_estimate_matrix.sum(axis = 1)
basic_estimate = basic_estimate.astype(int)

training = training.reset_index()

basic_estimate_log = np.log(basic_estimate / training.prom)
basic_estimate_residual = training.log_price_plus_comdebt - basic_estimate_log

training = training.assign(basic_estimate = basic_estimate)
training = training.assign(basic_estimate_log = basic_estimate_log)
training = training.assign(basic_estimate_residual = basic_estimate_residual)

training = training.assign(basic_estimate_deviation = training["Total price"] - training.basic_estimate)
alva_io.write_to_csv(training, "C:/Users/tobiasrp/data/basic_estimate_residuals.csv")

print "Constructing basic estimates for test set"


# ------------
# Construct basic estimates for test set
# ------------


test = test.assign(reg_prediction = reg_predictions) # add regression prediction (y_reg) to test set
test = test.assign(reg_prediction_nat = (np.exp(reg_predictions) * test.prom).astype(int))

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
basic_est_1 = 0.5 * test.reg_prediction_nat + 0.5 * test.prev_sale_estimate_1
basic_est_2 = 0.5 * test.reg_prediction_nat + 0.5 * (0.9 * test.prev_sale_estimate_1 + 0.1 * test.prev_sale_estimate_2)
basic_est_3 = 0.5 * test.reg_prediction_nat + 0.5 * (0.85 * test.prev_sale_estimate_1 + 0.15 * (0.8 * test.prev_sale_estimate_2 + 0.2 * test.prev_sale_estimate_3))

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

# ----------------
# Simple comparable implementation
# ----------------

count = 0.0
size = float(len(test.index))
old_progress, new_progress = 0,0
resid_median_list = []

print ""
print "Mapping distances for comparable method"
print "0%",

mapping_start = time.time()

for index, r in test.iterrows():
    district_matrix = training.loc[training.bydel_code == r.bydel_code] #HUSK Å NOTERE DETTE ET STED
    district_matrix = district_matrix.append(r)

    distance_matrix = squareform(pdist(district_matrix[["coord_x","coord_y"]]))

    max_distance = 0.000025
    prev_max_distance = -1
    close_ids = []

    while len(close_ids) < COMPARABLE_SET_SIZE and max_distance <= 0.0025:
        close_points = (distance_matrix[-1] <= max_distance) & (distance_matrix[-1] > prev_max_distance)
        close_points[-1] = False # exclude itself
        close_ids += [district_matrix.iloc[index].name for index, close in enumerate(close_points) if close]
        prev_max_distance = max_distance
        max_distance *= 1.1

    close_ids = close_ids[:COMPARABLE_SET_SIZE]

    close_resids = training.loc[close_ids].basic_estimate_residual

    resid_median = close_resids.median()

    if len(close_resids < 3):
        resid_median /= 2

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
resid_median_column = pd.Series(resid_median_list)

#test = test.reset_index()

test = test.assign(residual_adjusted_basic_estimate = test.basic_estimate_log + resid_median_column)
test = test.assign(residual_adjusted_basic_estimate = test.residual_adjusted_basic_estimate.fillna(test.basic_estimate_log))

test = test.assign(residual_adjusted_basic_estimate_nat = (test.prom * np.exp(test.residual_adjusted_basic_estimate)).astype(int))

### LATER, CALCULATE ERROR FOR REGRESSION AND RESIDUAL ADJUSTED ESTIMATES

alva_io.write_to_csv(test,"C:/Users/tobiasrp/data/residual_adjusted_estimates.csv")

# END COMPARABLE MODEL


score = pd.DataFrame()


score = score.assign(true_value = test["Total price"])

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

score = score.assign(comparable_prediction = test.residual_adjusted_basic_estimate_nat)
score = score.assign(comparable_deviation = score.true_value - score.comparable_prediction)
score = score.assign(comparable_deviation_percentage = abs(score.comparable_deviation / score.true_value))

print ""
print " -------- Test Results -------- "
print ""
print "Medianfeil regresjon:",100*round(score.reg_deviation_percentage.median(),5),"%"
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
print "Basic-estimat, medianfeil:",100*round(score.basic_deviation_percentage.median(),5),"%"
print score.basic_deviation_percentage.quantile([.25, .5, .75])


print "Comparable-estimat, medianfeil:",100*round(score.comparable_deviation_percentage.median(),5),"%"
print score.comparable_deviation_percentage.quantile([.25, .5, .75])


# fig = sm.graphics.plot_partregress("sqmeter_price","prom",["sold_month","ordinal_sold_date","bydel"],data=subset, obs_labels=False)
# fig.show()