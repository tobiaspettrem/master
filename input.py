import pandas as pd, numpy as np, time
virdi_excel = pd.ExcelFile("C:/Users/tobiasrp/data/20180220 Transactions Virdi v2.xlsx")

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
address = pd.read_csv("C:/Users/tobiasrp/data/20180223 Address table v2.csv", sep=";", dtype=types, usecols=[0, 36, 37, 85, 86])
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

print("Dropping duplicates")
merge = merge.drop_duplicates(subset="ad_code", keep=False)

print(merge.head())
print(merge.info())
print(merge.isnull().sum())


pd.DataFrame.to_csv(merge,"C:/Users/tobiasrp/data/merge.csv")

'''

# ------------------
#   DO REGRESSION
# ------------------

y,X = dmatrices('log_price_plus_comdebt ~ C(size_group,Treatment(reference="40 - 49")) + \
sold_month + C(bydel,Treatment(reference="FROGNER")) + year_group + bygningstype', \
                data=training, return_type = "dataframe")

#describe model
print("y")
print(y.head())
print("x")
print(X.head())
print("---")

mod = sm.OLS(y,X,missing = 'drop')

#fit model
res = mod.fit()

# resids = res.resid
# plt.scatter(resids.index,resids)
# plt.show()

#magic
print(res.summary())
print("-----------------")
print("Robust standard error")
print()
print(res.HC0_se)

# exogen_matrix = [c for c in subset.columns if c not in regression_columns]

y_test,X_test = dmatrices('log_price_plus_comdebt ~ C(size_group,Treatment(reference="40 - 49")) + \
sold_month + C(bydel,Treatment(reference="FROGNER")) + year_group + bygningstype', \
                data=test, return_type = "dataframe")


test_results = res.predict(X_test)

print("Test Results")

score = pd.DataFrame()

score = score.assign(predicted_value = (test.prom*np.exp(test_results) - test.common_debt)).astype(int)
score = score.assign(true_value = test.sold_price)
score = score.assign(deviation_absolute = score.true_value - score.predicted_value)
score = score.assign(deviation_percentage = abs(score.deviation_absolute / score.true_value))

print(score.head(20))
print("Median feil:",100*round(score.deviation_percentage.median(),5),"%")
print()
print("Gjennomsnitt feil:",100*round(score.deviation_percentage.mean(),5),"%")
print()
print("Kvantiler")
print(score.quantile([0.25,0.5,0.75]))

# fig = sm.graphics.plot_partregress("sqmeter_price","prom",["sold_month","ordinal_sold_date","bydel"],data=subset, obs_labels=False)
# fig.show()
# time.sleep(10)
'''