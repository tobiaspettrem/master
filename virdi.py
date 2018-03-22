import alva_io, address, pandas as pd, numpy as np

virdi_df = alva_io.get_dataframe_from_excel("C:/Users/tobiasrp/data/20180220 Transactions Virdi v2.xlsx")

address_df = address.get_address()

virdi_augmented = pd.merge(virdi_df, address_df, on="id")

print(virdi_augmented.head())
print(virdi_augmented.info())
print(virdi_augmented.isnull().sum())

print virdi_augmented.shape
print("Dropping duplicates")
virdi_augmented = virdi_augmented.drop_duplicates(subset="ad_code", keep=False)
print virdi_augmented.shape

print(virdi_augmented.head())
print(virdi_augmented.info())
print(virdi_augmented.isnull().sum())

pd.DataFrame.to_csv(virdi_augmented, "C:/Users/tobiasrp/data/virdi_augmented.csv")

# virdi = virdi.loc[virdi["kr/m2"] > ] # cutting on sqm price

virdi_augmented = virdi_augmented.assign(log_price_plus_comdebt = np.log(virdi_augmented["kr/m2"]))

# ----- CREATE NEW COLUMN size_group ------ #

size_group_size = 10 #change this only to change group sizes
size_labels = ["10 - 29"]
size_labels += [ "{0} - {1}".format(i, i + size_group_size - 1) for i in range(30, 150, size_group_size) ]
size_labels.append("150 - 179")
size_labels.append("180 - ")

size_thresholds = [10]
size_thresholds += [i for i in range(30,151,size_group_size)] + [180] + [500]

virdi_augmented['size_group'] = pd.cut(virdi_augmented.prom, size_thresholds, right=False, labels=size_labels)

print virdi_augmented.head(10)


# ------------------
#   DO REGRESSION
# ------------------

virdi_augmented = virdi_augmented.sample(frac=1) # shuffle the dataset to do random training and test partition

threshold = int(0.8 * len(virdi_augmented))
training = virdi_augmented[:threshold]
test = virdi_augmented[threshold:]

y,X = dmatrices('log_price_plus_comdebt ~ C(size_group,Treatment(reference="40 - 49")) + \
sold_month + C(bydel_code,Treatment(reference="bfr")) + unit_type', \
                data=training, return_type = "dataframe") # excluding build year until videre

"""

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

"""