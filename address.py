import alva_io, numpy as np, add_district

def get_address():
    types = {'id': np.dtype(int),
             'bydel': str,
             'bydelsnr': np.dtype(float),
             'coord_x': np.dtype(float),
             'coord_y': np.dtype(float),
             'eiendomadressebokstav': str,
             'postnr': np.dtype(int),
             'poststed': str,
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