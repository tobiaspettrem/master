import alva_io, numpy as np, add_district, pandas as pd, time

def get_address():
    types = {'id': np.dtype(int),
             'bydel': str,
             'bydelsnr': np.dtype(float),
             'coord_x': np.dtype(float),
             'coord_y': np.dtype(float),
             'eiendomadressebokstav': str,
             'postnr': np.dtype(int),
             'poststed': str,
             'borettslagetsnavn': str,
             'unit_type': str,
             'eiendomadressegatenavn': str,
             'eiendomadressehusnr': np.dtype(float)} # use float instead of int when NaN values are present. Pandas unable to handle NaN as int

    address = alva_io.get_dataframe_from_csv("C:/Users/tobiasrp/data/20180223 Address table v2.csv", ";", types, types.keys()) # load dataframe from file

    address = address.loc[address.poststed == "OSLO"] # discard data on all houses outside Oslo

    address = add_district.add_bydel(address) # retrieve district info from Oslo homepage

    alva_io.write_to_csv(address,"C:/Users/tobiasrp/data/address_with_districts.csv")

    print address.shape
    address = address.dropna(subset = ["bydel_code"])
    print address.shape

    return address

def get_previous_sales():

    types = {'address_id': np.dtype(int),
             'source': str,
             'is_approved': str,
             'official_price': np.dtype(float),
             'official_date': str}

    grunnbok = alva_io.get_dataframe_from_csv("C:/Users/tobiasrp/data/20180223 Grunnbok table v2.csv", ";", types, types.keys()) # load dataframe from file
    #grunnbok = grunnbok.loc[grunnbok.poststed == "OSLO"] # discard data on all houses outside Oslo

    before_notapproved_dropping = grunnbok.shape[0]
    grunnbok = grunnbok.loc[grunnbok.is_approved == 't']
    print "Dropped not approved columns, amounted to " + str (before_notapproved_dropping - grunnbok.shape[0]) + " rows or " + str(round(100*((before_notapproved_dropping - float(grunnbok.shape[0])) / before_notapproved_dropping),2)) + "%."

    before_source_dropping = grunnbok.shape[0]
    grunnbok = grunnbok.loc[grunnbok.source == 'ambita']
    print "Dropped columns from csv or FINN, amounted to " + str (before_source_dropping - grunnbok.shape[0]) + " rows or " + str(round(100*((before_source_dropping - float(grunnbok.shape[0])) / before_source_dropping),2)) + "%."

    before_price_dropping = grunnbok.shape[0]
    MAX_PRICE = 50000
    grunnbok = grunnbok.loc[grunnbok.official_price > MAX_PRICE]
    print "Dropped rows under " + str(MAX_PRICE) + ", amounted to " + str (before_price_dropping - grunnbok.shape[0]) + " rows or " + str(round(100*((before_price_dropping - float(grunnbok.shape[0])) / before_price_dropping),2)) + "%."

    grunnbok = grunnbok.assign(official_date=pd.to_datetime(grunnbok["official_date"], format="%Y-%m-%d"))
    grunnbok = grunnbok.assign(sold_year = grunnbok.official_date.map(lambda x: x.year))

    before_year_dropping = grunnbok.shape[0]
    grunnbok = grunnbok.loc[grunnbok.sold_year >= 1993]
    print "Columns with sales date too old amounted to " + str(before_year_dropping - grunnbok.shape[0]) + " rows or " + str(round(100*((before_year_dropping - float(grunnbok.shape[0])) / before_year_dropping),2)) + "%."

    print ""

    grunnbok = grunnbok.sort_values(by=['address_id', 'official_date'], ascending=[True,False])

    grunnbok = grunnbok.rename(columns = {'address_id':'id'})

    return grunnbok

    """
    ### OLD, FROM ADDRESS:
    before_na_dropping = grunnbok.shape[0]
    grunnbok = grunnbok.dropna()
    print "Dropped NA columns, amounted to " + str (before_na_dropping - grunnbok.shape[0]) + " rows or " + str(round(100*((before_na_dropping - float(grunnbok.shape[0])) / before_na_dropping),2)) + "%."

    before_null_dropping = grunnbok.shape[0]
    grunnbok = grunnbok.loc[grunnbok.official_date != "0001-01-01"]
    print "Columns with invalid datetime amounted to " + str (before_null_dropping - grunnbok.shape[0]) + " rows or " + str(round(100*((before_null_dropping - float(grunnbok.shape[0])) / before_null_dropping),2)) + "%."
    """
