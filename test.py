import pandas as pd, numpy as np

price_index_ssb = pd.read_csv("C:/Users/tobiasrp/data/price_index_ssb.csv",sep=";")

print price_index_ssb.head()

def get_quarter(date):
    if pd.isnull(date):
        return np.nan
    print "DATE: " + str(date)
    year = date.year
    month = date.month
    print "YEAR: " + str(year)
    print "MONTH: " + str(month)
    quarter = 1 + ((month - 1) // 3)
    quarter_string = str(year) + "_Q" + str(quarter)
    print "OFFICIAL QUARTER: " + quarter_string
    return quarter_string

virdi_augmented = pd.read_csv("C:/Users/tobiasrp/data/virdi_augmented_with_title.csv",index_col=0)

#virdi_augmented = virdi_augmented.assign(official_date=virdi_augmented.official_date.map(lambda x: pd.to_datetime(x)))

#virdi_augmented = virdi_augmented.assign(official_quarter = virdi_augmented.official_date.map(lambda x: get_quarter(x)))