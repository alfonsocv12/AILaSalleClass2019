import numpy as np, pandas as pd, plotly, plotly.plotly as py, plotly.graph_objs as go, matplotlib.pyplot as plt
import json

movies = pd.read_csv('train.csv',index_col=None, na_values=['NA'])

#10 valores csv
print(movies.head(10))
#tipos de valores
print(movies.dtypes)
print('')
#datos estadisticos de valores nominales
print(movies.describe())
print('')
#nos da el pocentaje de datos nulos
print(pd.DataFrame({'percent_missing': movies.isnull().sum() * 100 / len(movies)}))
print('')
#Nos dice que tan impotates son los datos unicos
print(pd.DataFrame({'percent_unique': movies.apply(lambda x: x.unique().size/x.size*100)}))

# print(movies.belongs_to_collection)

# for movie in movies:
#     print(movie)
