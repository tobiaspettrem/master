import pandas as pd, numpy as np, time
virdi_excel = pd.ExcelFile("data/20180220 Transactions Virdi v2.xlsx")

virdi = virdi_excel.parse("Query result")

print(virdi.head())
print(virdi.shape)
print(virdi.isnull().sum())
print("-------")

types = {'id': np.dtype(float),
         'coord_x': np.dtype(float),
         'coord_y': np.dtype(float),
         'eiendomadressegatenavn': str,
         'eiendomadressehusnr': np.dtype(float)}

start = time.time()
print("Reading address")
address = pd.read_csv("data/20180223 Address table v2.csv", sep=";", dtype=types, usecols=[0, 36, 37, 85, 86])
print("Finished reading address. " + str(round(time.time() - start)) + " seconds elapsed.")
print("-------")
"""
print(address.head())
print(address.info())
print(address.isnull().sum())
"""
merge = pd.merge(virdi,address,on="id")

print(merge.head())
print(merge.info())
print(merge.isnull().sum())

pd.DataFrame.to_csv(merge,"data/merge.csv")