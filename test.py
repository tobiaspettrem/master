import alva_io, numpy as np, pandas as pd

test = pd.read_csv("C:/Users/tobiasrp/data/address_with_districts.csv")

print test.shape

print test.isnull().sum()