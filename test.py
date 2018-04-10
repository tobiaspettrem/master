
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
COMPARABLE_SET_SIZE = 10

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

else:
    virdi_augmented = pd.read_csv("C:/Users/tobiasrp/data/virdi_augmented_with_title.csv",index_col=0)

if raw_input("Enter to run without mapping previous sales") != "":
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
    print virdi_augmented.bydel_code.value_counts()
    virdi_augmented.loc[virdi_augmented.bydel_code == 'SEN', 'bydel_code'] = "bsh" # reassign SENTRUM to St. Hanshaugen
    virdi_augmented = virdi_augmented.loc[virdi_augmented.bydel_code != "MAR"]
    print virdi_augmented.bydel_code.value_counts()


    print virdi_augmented.sold_month_and_year.value_counts()
    virdi_augmented = virdi_augmented.loc[virdi_augmented.sold_month_and_year != "Feb_2018"]
    virdi_augmented = virdi_augmented.loc[virdi_augmented.sold_month_and_year != "Jan_2018"]
    print virdi_augmented.sold_month_and_year.value_counts()
