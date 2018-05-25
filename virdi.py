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

if raw_input("Enter to run without new initialization") != "":

    virdi_df = alva_io.get_dataframe_from_excel("C:/Users/" + USER_STRING + "/data/20180220 Transactions Virdi v2.xlsx")
    print "print1", virdi_df.shape

    address_df = address.get_address()

    virdi_augmented = pd.merge(virdi_df, address_df, on="id")

    print("Dropping duplicates")
    print "print1", virdi_augmented.shape
    virdi_augmented = virdi_augmented.drop_duplicates(subset="ad_code", keep=False)
    print "print2", virdi_augmented.shape

    print("Dropping NaN in ad_code")
    print "print1", virdi_augmented.shape
    virdi_augmented = virdi_augmented.dropna(subset = ["ad_code"])
    print "print2", virdi_augmented.shape

    pd.DataFrame.to_csv(virdi_augmented, "C:/Users/" + USER_STRING + "/data/virdi_augmented.csv")
    #after these operations, scrape FINN for additional data to create virdi augmented with title

if raw_input("Enter to run without mapping previous sales") != "":

    virdi_augmented = pd.read_csv("C:/Users/" + USER_STRING + "/data/virdi_augmented_with_title.csv",index_col=0)

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

    #PERHAPS DANGER HERE: CHECK
    print len(virdi_augmented.index)
    print virdi_augmented.tail()

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
                count_adjustment += 1
            else:
                virdi_augmented[prev_sale_price_col_name][index - count] = row.prev_sale_price_1
                virdi_augmented[prev_sale_date_col_name][index - count] = row.prev_sale_date_1
                count_adjustment = 0
            delete_rows.append(index)
        elif id != prev_id:
            if not pd.isnull(row.prev_sale_price_1):
                count_adjustment = -1
            else:
                count_adjustment = 0
            count = 0
        else:
            delete_rows.append(index)

        new_progress = round(progress_count / size, 2)
        if old_progress != new_progress:
            if (int(100 * new_progress)) % 10 == 0:
                print str(int(100 * new_progress)) + "%",
            else:
                print "|",
        old_progress = new_progress

        progress_count += 1

    virdi_augmented = virdi_augmented.drop(virdi_augmented.index[delete_rows])
    alva_io.write_to_csv(virdi_augmented,"C:/Users/" + USER_STRING + "/data/virdi_aug_title_prev_sale.csv")

    print ""
    print ""

    #########################

else:
    virdi_augmented = pd.read_csv("C:/Users/" + USER_STRING + "/data/virdi_aug_title_prev_sale_correct.csv", sep = ";")
    virdi_augmented = virdi_augmented.assign(real_sold_date = pd.to_datetime(virdi_augmented["real_sold_date"],format="%d.%m.%Y"))

print virdi_augmented.shape
wp_bool = ""

while wp_bool not in ["y","n"]:
    wp_bool = raw_input("Run autoregressive? y or n: ")

if wp_bool == "y":
    wp_bool = True
else:
    wp_bool = False

kmeans_bool = ""
while kmeans_bool not in ["y","n"]:
    kmeans_bool = raw_input("Run k-means to generate districts? y or n: ")

if kmeans_bool == "y":
    kmeans_bool = True
    KMEANS_K = int(raw_input("K: "))
    print "Running K-means with k = " + str(KMEANS_K)
else:
    kmeans_bool = False


moran_and_geary_bool = ""
while moran_and_geary_bool not in ["y","n"]:
    moran_and_geary_bool = raw_input("Calculate Geary's C and Moran's I? y or n: ")

if moran_and_geary_bool == "y":
    moran_and_geary_bool = True
else:
    moran_and_geary_bool = False

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

virdi_augmented = virdi_augmented.assign(title_lower = virdi_augmented.title.str.lower())

virdi_augmented = virdi_augmented.assign(needs_refurbishment = 0)
virdi_augmented.needs_refurbishment = virdi_augmented.title_lower.str.contains("oppussingsobjekt")
virdi_augmented.needs_refurbishment = virdi_augmented.needs_refurbishment | virdi_augmented.title_lower.str.contains("oppgraderingsbehov")
virdi_augmented.needs_refurbishment = virdi_augmented.needs_refurbishment | virdi_augmented.title_lower.str.contains("oppussingsbehov")
virdi_augmented.needs_refurbishment = virdi_augmented.needs_refurbishment | virdi_augmented.title_lower.str.contains("modernisering")
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

##----------------------##
## AUTOREGRESSIVE MODEL ##
##----------------------##

#START

virdi_augmented = virdi_augmented.sample(frac=1) # shuffle the dataset to do random training and test partition
virdi_augmented = virdi_augmented.reset_index(drop=True)

threshold = int(TRAINING_SET_SIZE * len(virdi_augmented))

training = virdi_augmented[:threshold]


if wp_bool:

    # BEGIN AUTOREGRESSIVE

    not_enough_adjacent_houses_count = 0.0
    count = 0.0
    size = float(len(virdi_augmented.index))
    old_progress, new_progress = 0,0
    WP_list = []

    print ""
    print "Running autoregressive model with size " + str(AUTOREGRESSIVE_COMPARABLE_SET_SIZE) + ". Mapping distances:"
    print "0%",
    for_loop_start = time.time()

    for index, r in virdi_augmented.iterrows():

        comparable_matrix = training.loc[training.prom > 0]
        comparable_matrix = comparable_matrix.loc[abs(comparable_matrix.coord_x - r.coord_x) + abs(comparable_matrix.coord_y - r.coord_y) < 0.01]

        comparable_matrix = comparable_matrix.loc[comparable_matrix.real_sold_date < r.real_sold_date] # comparable sale must have occured earlier in time

        comparable_matrix = comparable_matrix.append(r)

        # --- new method
        comparable_coords = comparable_matrix[["coord_x", "coord_y"]].as_matrix()
        N = len(comparable_coords)
        distance_matrix = np.zeros(N)
        loni, lati = comparable_coords[-1]
        for j in xrange(N):
            lonj, latj = comparable_coords[j]
            distance_matrix[j] = kmeans.coord_distance(lati, loni, latj, lonj)

        max_distance = 0.001  # in km, so 1 meter
        prev_max_distance = -1
        close_ids = []

        while len(close_ids) < AUTOREGRESSIVE_COMPARABLE_SET_SIZE and max_distance <= 1.5:

            close_points = (distance_matrix <= max_distance) & (distance_matrix > prev_max_distance)
            if max_distance == 0.001:
                close_points[-1] = False  # exclude itself
            close_ids += [i for i, close in enumerate(close_points) if close]
            prev_max_distance = max_distance
            max_distance *= 1.1

        close_ids = close_ids[:AUTOREGRESSIVE_COMPARABLE_SET_SIZE]
        close_ids_comparable_matrix = [comparable_matrix.iloc[i].name for i in close_ids]

        """
        # OLD METHOD
        distance_matrix = squareform(pdist(comparable_matrix[["coord_x", "coord_y"]]))

        max_distance = 0.000025 # about 1.5 meters
        prev_max_distance = -1
        close_ids = []
        close_ids_comparable_matrix = []

        while len(close_ids) < AUTOREGRESSIVE_COMPARABLE_SET_SIZE + 1 and max_distance <= 0.025: # about 1500 meters
            close_points = (distance_matrix[-1] <= max_distance) & (distance_matrix[-1] > prev_max_distance)
            if max_distance == 0.000025:
                close_points[-1] = False  # exclude itself - if present, always the last element
            close_ids_comparable_matrix += [comparable_matrix.iloc[i].name for i, close in enumerate(close_points) if close]
            close_ids += [i for i, close in enumerate(close_points) if close]
            prev_max_distance = max_distance
            max_distance *= 1.5

        close_ids_comparable_matrix = close_ids_comparable_matrix[:AUTOREGRESSIVE_COMPARABLE_SET_SIZE]
        close_ids = close_ids[:AUTOREGRESSIVE_COMPARABLE_SET_SIZE]
        
        """

        distances = distance_matrix[[close_ids]]
        if len(distances) > 0:
            if distances.max() == 0:
                distances[distances == 0] = 0.000025 # value is irrelevant
            else:
                distances[distances == 0] = distances[distances > 0].min()

        distances = 1 / distances

        p_times_distance = distances * comparable_matrix.loc[close_ids_comparable_matrix].log_price_plus_comdebt

        WP = p_times_distance.sum() / distances.sum()

        if len(close_ids) < 1:
            WP = training.loc[training.bydel_code == r.bydel_code].log_price_plus_comdebt.mean()
            not_enough_adjacent_houses_count += 1

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

    print "2"
    print virdi_augmented.shape

    print ""
    print "Done. " + str(round(time.time() - for_loop_start )) + " seconds elapsed."
    WP_column = pd.Series(WP_list)

    print "Virdi augmented with autoregressive terms."
    print "If number of nearby houses lower than 1:"
    print "Set autoregressive term to mean of all prices: " + str(virdi_augmented.log_price_plus_comdebt.mean())

    print "Number of houses with lower than 1 nearby: " + str(not_enough_adjacent_houses_count)
    print "which amounts to " + str(100*round(not_enough_adjacent_houses_count / len(virdi_augmented),4)) + "% of data set."

    virdi_augmented = virdi_augmented.assign(WP = WP_column)
    virdi_augmented = virdi_augmented.dropna(subset = ["WP"])

    alva_io.write_to_csv(virdi_augmented,"C:/Users/" + USER_STRING + "/data/virdi_augmented_with_WP.csv")

    # END AUTOREGRESSIVE

# ------------
# Calculate repeat sales estimates
# ------------

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

price_index_ssb = pd.read_csv("C:/Users/" + USER_STRING + "/data/price_index_oslo_ssb.csv", sep=";")

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

print "Data set shape before regression:"
print virdi_augmented.shape


alva_io.write_to_csv(virdi_augmented,"C:/Users/" + USER_STRING + "/data/virdi_aug_title_prev_sale_estimates.csv")

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

    #kmeans_district_map.plot_districts(training, "bydel_code", 0)
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

y,X = dmatrices('log_price_plus_comdebt ~ C(size_group,Treatment(reference="40 -- 49")) + \
C(sold_month_and_year,Treatment(reference="Apr_2017")) + C(' + district_string_train +') + \
C(unit_type,Treatment(reference="house")) + needs_refurbishment + year_group + has_garden + \
is_borettslag + is_penthouse + has_terrace + common_cost_is_high + has_two_bedrooms + \
has_three_bedrooms' + wp_string, data=training, return_type = "dataframe")


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
print res.summary()

"""
### ROBUST STANDARD ERROR
print ""
print("Robust standard error")
print(res.HC0_se)
"""

y_test,X_test = dmatrices('log_price_plus_comdebt ~ C(size_group,Treatment(reference="40 -- 49")) + \
C(sold_month_and_year,Treatment(reference="Apr_2017")) + C(' + district_string_test +') + \
C(unit_type,Treatment(reference="house")) + needs_refurbishment + year_group + has_garden + \
is_borettslag + is_penthouse + has_terrace + common_cost_is_high + has_two_bedrooms + \
has_three_bedrooms' + wp_string, data=test, return_type = "dataframe")

reg_predictions = res.predict(X_test)

print "Regression done"
print str(round(time.time() - reg_start,4)) + "seconds elapsed"
# ------------
# Regression done
# ------------

training = training.assign(fitted_values = res.fittedvalues)
training = training.assign(fitted_values_nat = (np.exp(res.fittedvalues) * training.prom).astype(int))
training = training.assign(regression_residuals = res.resid)

test = test.assign(reg_prediction = reg_predictions) # add regression prediction (y_reg) to test set
test = test.assign(reg_prediction_nat = (np.exp(reg_predictions) * test.prom).astype(int))
test = test.assign(regression_residual = test.log_price_plus_comdebt - reg_predictions)

alva_io.write_to_csv(training,"C:/Users/" + USER_STRING + "/data/training" + str(RUN_NUMBER) + ".csv")
alva_io.write_to_csv(test,"C:/Users/" + USER_STRING + "/data/test" + str(RUN_NUMBER) + ".csv")


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


### Count number of previous sales, test set

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

    if kmeans_bool:
        district_matrix = training.loc[training.kmeans_cluster == r.kmeans_cluster_prediction] #HUSK Å NOTERE DETTE ET STED
    else:
        district_matrix = training.loc[training.bydel_code == r.bydel_code] #HUSK Å NOTERE DETTE ET STED

    district_matrix = district_matrix.loc[district_matrix.real_sold_date < r.real_sold_date] # comparable sale must have occured earlier in time

    district_matrix = district_matrix.append(r)

    #distance_matrix = squareform(pdist(district_matrix[["coord_x","coord_y"]]))
    """
    p = pdist(district_matrix[["coord_x", "coord_y"]])

    print "p"
    print p

    print "--------"

    distance_matrix = squareform(p)

    print distance_matrix

    """
    comparable_coords = district_matrix[["coord_x", "coord_y"]].as_matrix()
    N = len(comparable_coords)
    distance_matrix = np.zeros(N)
    loni, lati = comparable_coords[-1]
    for j in xrange(N):
        lonj, latj = comparable_coords[j]
        distance_matrix[j] = kmeans.coord_distance(lati, loni, latj, lonj)

    max_distance = 0.001 # in km, so 1 meter
    prev_max_distance = -1
    close_indexes = []

    while len(close_indexes) < COMPARABLE_SET_SIZE and max_distance <= 0.15:

        close_points = (distance_matrix <= max_distance) & (distance_matrix > prev_max_distance)
        if max_distance == 0.001:
            close_points[-1] = False # exclude itself
        close_indexes += [index for index, close in enumerate(close_points) if close]
        prev_max_distance = max_distance
        max_distance *= 1.1

    close_indexes = close_indexes[:COMPARABLE_SET_SIZE]

    close_ids = [district_matrix.iloc[index].name for index in close_indexes]

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

"""
# ------------
# Construct hybrid estimates for training set
# ------------

print ""
print "Constructing basic estimates for training set"

number_of_prev_sales_dummy = pd.get_dummies(training.number_of_prev_sales)

basic_est_0 = training.fitted_values_nat
basic_est_1 = REG_SHARE * training.fitted_values_nat + (1 - REG_SHARE) * training.prev_sale_estimate_1
basic_est_2 = REG_SHARE * training.fitted_values_nat + (1 - REG_SHARE) * (0.9 * training.prev_sale_estimate_1 + 0.1 * training.prev_sale_estimate_2)
basic_est_3 = REG_SHARE * training.fitted_values_nat + (1 - REG_SHARE) * (0.85 * training.prev_sale_estimate_1 + 0.15 * (0.8 * training.prev_sale_estimate_2 + 0.2 * training.prev_sale_estimate_3))

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
alva_io.write_to_csv(training, "C:/Users/" + USER_STRING + "/data/basic_estimate_residuals.csv")

"""

print ""
print "Constructing hybrid estimates for test set"

# ------------
# Construct basic estimates for test set
# ------------

#print test.sort_values(by = ["regression_residual"], ascending=True).head(20)


### Remove repeated estimates if it deviates too much from fitted value (test set)

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
    print "K-means price factor: " + str(PRICE_FACTOR)
else:
    print "Used administrative districts"
print "Reg share: " + str(REG_SHARE)