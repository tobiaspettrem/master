from multiprocessing import Pool
import time
import scrap_finn
from scipy.spatial.distance import pdist, squareform
import alva_io, address
import pandas as pd
import numpy as np
import calendar
from patsy import dmatrices
import statsmodels.api as sm
import matplotlib.pyplot as plt

SOLD_TO_OFFICIAL_DIFF = 55
TRAINING_SET_SIZE = 0.9
COMPARABLE_SET_SIZE = 10

if raw_input("Enter to run without new initialization") != "":

    virdi_df = alva_io.get_dataframe_from_excel("C:/Users/Tobias/data/20180220 Transactions Virdi v2.xlsx")

    address_df = address.get_address()

    virdi_augmented = pd.merge(virdi_df, address_df, on="id")

    print("Dropping duplicates")
    virdi_augmented = virdi_augmented.drop_duplicates(subset="ad_code", keep=False)

    print("Dropping NaN in ad_code")
    print "print1", virdi_augmented.shape
    virdi_augmented = virdi_augmented.dropna(subset = ["ad_code"])
    print "print2", virdi_augmented.shape

    pd.DataFrame.to_csv(virdi_augmented, "C:/Users/Tobias/data/virdi_augmented.csv")

else:
    virdi_augmented = pd.read_csv("C:/Users/Tobias/data/virdi_augmented_with_title.csv",index_col=0)


# virdi = virdi.loc[virdi["kr/m2"] > ] # cutting on sqm price

virdi_augmented = virdi_augmented.assign(log_price_plus_comdebt = np.log(virdi_augmented["kr/m2"]))

# ----- CREATE NEW COLUMN size_group ------ #

size_group_size = 10 #change this only to change group sizes
size_labels = ["10 -- 29"]
size_labels += [ "{0} - {1}".format(i, i + size_group_size - 1) for i in range(30, 150, size_group_size) ]
size_labels.append("150 - 179")
size_labels.append("180 - ")

size_thresholds = [10]
size_thresholds += [i for i in range(30,151,size_group_size)] + [180] + [500]

virdi_augmented['size_group'] = pd.cut(virdi_augmented.prom, size_thresholds, right=False, labels=size_labels)

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

virdi_augmented = virdi_augmented.sample(frac=1) # shuffle the dataset to do random training and test partition
# virdi_augmented = virdi_augmented[["log_price_plus_comdebt","size_group","sold_month_and_year","bydel_code","unit_type", "prom","Total price","coord_x","coord_y"]]

columns_to_count = ["size_group","sold_month_and_year","bydel_code","unit_type"]

# virdi_augmented = virdi_augmented.loc[virdi_augmented.bydel_code != "SEN"]
virdi_augmented.loc[virdi_augmented.bydel_code == 'SEN', 'bydel_code'] = "bsh" # reassign SENTRUM to St. Hanshaugen

virdi_augmented = virdi_augmented.loc[virdi_augmented.bydel_code != "MAR"]

virdi_augmented = virdi_augmented.loc[virdi_augmented.sold_month_and_year != "Feb_2018"]
virdi_augmented = virdi_augmented.loc[virdi_augmented.sold_month_and_year != "Jan_2018"]
virdi_augmented = virdi_augmented.loc[virdi_augmented.unit_type != "other"]

virdi_augmented = virdi_augmented.assign(needs_refurbishment = 0)
virdi_augmented = virdi_augmented.assign(title_lower = virdi_augmented.title.str.lower())
print virdi_augmented.needs_refurbishment.value_counts()
virdi_augmented.needs_refurbishment = virdi_augmented.title_lower.str.contains("oppussingsobjekt")
print virdi_augmented.needs_refurbishment.value_counts()
virdi_augmented.needs_refurbishment = virdi_augmented.needs_refurbishment | virdi_augmented.title_lower.str.contains("oppgraderingsbehov")
print virdi_augmented.needs_refurbishment.value_counts()
virdi_augmented.needs_refurbishment = virdi_augmented.needs_refurbishment | virdi_augmented.title_lower.str.contains("oppussingsbehov")
print virdi_augmented.needs_refurbishment.value_counts()

virdi_augmented.needs_refurbishment = virdi_augmented.needs_refurbishment.astype(int)

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
# begin splitting and regressing
# ------------

threshold = int(TRAINING_SET_SIZE * len(virdi_augmented))
training = virdi_augmented[:threshold]
test = virdi_augmented[threshold:]

y,X = dmatrices('log_price_plus_comdebt ~ C(size_group,Treatment(reference="40 - 49")) + \
C(sold_month_and_year,Treatment(reference="Apr_2017")) + C(bydel_code,Treatment(reference="bfr")) + \
C(unit_type,Treatment(reference="house")) + needs_refurbishment', \
                data=training, return_type = "dataframe") # excluding build year until videre


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

resids = res.resid

virdi_with_resids = training
virdi_with_resids["resids"] = resids

alva_io.write_to_csv(virdi_with_resids,"C:/Users/Tobias/data/virdi_with_resids.csv")
"""
"""
#magic
print(res.summary())
print("-----------------")
print("Robust standard error")
print()
print(res.HC0_se)

#plt.scatter(resids.index,resids)
#plt.show()

# exogen_matrix = [c for c in subset.columns if c not in regression_columns]

y_test,X_test = dmatrices('log_price_plus_comdebt ~ C(size_group,Treatment(reference="40 - 49")) + \
C(sold_month_and_year,Treatment(reference="Apr_2017")) + C(bydel_code,Treatment(reference="bfr")) + \
C(unit_type,Treatment(reference="house")) + needs_refurbishment', \
                data=test, return_type = "dataframe")


test_results = res.predict(X_test)

time.sleep(2)

# ----------------
# from here: simple comparable implementation
# ----------------
"""
test = test.assign(regression_prediction = test_results) # add regression prediction (y_reg) to test set

count = 0.0
size = float(len(test.index))
old_progress, new_progress = 0,0
resid_median_list = []

print "Mapping distances"
print "0%",

for index, r in test.iterrows():
    district_matrix = training.loc[training.bydel_code == r.bydel_code]
    district_matrix = district_matrix.append(r)

    distance_matrix = squareform(pdist(district_matrix[["coord_x","coord_y"]]))

    max_distance = 0.000025

    close_ids = []

    while len(close_ids) < COMPARABLE_SET_SIZE and max_distance <= 0.0025:
        close_points = (distance_matrix[-1] < max_distance)
        close_points[-1] = False # exclude itself
        close_ids = [district_matrix.iloc[index].name for index, close in enumerate(close_points) if close]
        max_distance *= 10

    close_ids = close_ids[:COMPARABLE_SET_SIZE]

    close_resids = training.loc[close_ids].resids

    #print close_ids
    resid_median = close_resids.median()

    resid_median_list.append(resid_median)

    new_progress = round(count / size,2)
    if old_progress != new_progress:
        if (int(100*new_progress)) % 10 == 0:
            print str(int(100*new_progress)) + "%",
        else:
            print "|",
    old_progress = new_progress

    # print count
    count += 1


resid_median_column = pd.Series(resid_median_list)

test = test.reset_index()

test = test.assign(residual_adjusted_prediction = test.regression_prediction + resid_median_column)

alva_io.write_to_csv(test,"C:/Users/tobiasrp/data/residual_adjusted_estimates.csv")
"""


score = pd.DataFrame()

score = score.assign(predicted_value = (test.prom*np.exp(test_results))).astype(int)
score = score.assign(true_value = test["Total price"])
score = score.assign(deviation = score.true_value - score.predicted_value)
score = score.assign(deviation_percentage = abs(score.deviation / score.true_value))

print ""
print "Test Results"
print ""
print(score.head())
print ""
print "Median feil:",100*round(score.deviation_percentage.median(),5),"%"
print ""
print "Gjennomsnitt feil:",100*round(score.deviation_percentage.mean(),5),"%"
print ""
print "Kvantiler"
print(score.quantile([0.25,0.5,0.75]))

# fig = sm.graphics.plot_partregress("sqmeter_price","prom",["sold_month","ordinal_sold_date","bydel"],data=subset, obs_labels=False)
# fig.show()