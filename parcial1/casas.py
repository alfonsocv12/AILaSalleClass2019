
#imports
import numpy as np, pandas as pd

# Read the file, in this case a csv using pandas, and the function will return a dataframe
casas_train = pd.read_csv('train.csv', index_col=None, na_values=['NA'])

print(casas_train.head(10))

print(casas_train.describe())

print(casas_train.dtypes)

print(casas_train.apply(lambda x: x.isnull().any()))

print(pd.DataFrame({'percent_missing': casas_train.isnull().sum() * 100 / len(casas_train)}))
