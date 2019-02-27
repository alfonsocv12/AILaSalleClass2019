
#imports
import numpy as np, pandas as pd, plotly, plotly.plotly as py, plotly.graph_objs as go, matplotlib.pyplot as plt, json, sys
from pctl_scale import PercentileScaler

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Read the file, in this case a csv using pandas, and the function will return a dataframe
casas_train = pd.read_csv('train.csv', index_col=None, na_values=['NA'])

#10 valores csv
print(casas_train.head(10))
#tipos de valores
print(casas_train.dtypes)
print('')
#datos estadisticos de valores nominales
print(casas_train.describe())
print('')
#nos da el pocentaje de datos nulos
print(pd.DataFrame({'percent_missing': casas_train.isnull().sum() * 100 / len(casas_train)}))
print('')
#Nos dice que tan impotates son los datos unicos
print(pd.DataFrame({'percent_unique': casas_train.apply(lambda x: x.unique().size/x.size*100)}))

plotly.tools.set_credentials_file(username='alfonsocv18', api_key='IukOlHfoQOc9CejJEThc')

target_class = casas_train.SalePrice.value_counts().values.tolist()

def plotLine(plotData, msg):

    data =[go.Bar(
        x=plotData.columns.values,
        y=plotData.index.values,
        name='LotArea'
        )]

    layout = dict(
        title = msg,
        xaxis= dict(title = plotData.columns.name),
        yaxis= dict(title= 'SalePrice'),
        barmode='group',
        autosize=False,
        width=800,
        height=500
        )

    fig = dict(data=data, layout=layout)
    #py.plot(fig)

print('')
#hipotesis 1
LotArea_data = pd.crosstab(casas_train.SalePrice, casas_train.LotArea)
print(LotArea_data)
plotLine(LotArea_data, 'Precio basado en LotArea')
#hipotesis 2
prueba = pd.crosstab(casas_train.SalePrice, casas_train.TotRmsAbvGrd)
print(prueba)
plotLine(prueba, 'Sale condition')

casas_train_limpio = casas_train[['MSSubClass', 'MSZoning', 'LotArea', 'Utilities', 'Condition1',
                                  'HouseStyle', 'OverallCond','ExterCond', 'BsmtCond', '1stFlrSF',
                                  '2ndFlrSF', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Functional','GarageArea',
                                  'GarageCond','PavedDrive','MiscVal', 'SaleType','SaleCondition',
                                  'SalePrice']]

print('')
print(casas_train_limpio.dtypes)
print('')
print(casas_train_limpio.describe())
print('')
print(pd.DataFrame({'percent_missing': casas_train_limpio.isnull().sum() * 100 / len(casas_train_limpio)}))

casas_train_limpio['MSZoning'] = pd.Categorical(casas_train_limpio.MSZoning)
casas_train_limpio['MSZoning'] = pd.get_dummies(casas_train_limpio['MSZoning'])
casas_train_limpio['Utilities'] = pd.Categorical(casas_train_limpio.Utilities)
casas_train_limpio['Utilities'] = pd.get_dummies(casas_train_limpio['Utilities'])
casas_train_limpio['Condition1'] = pd.Categorical(casas_train_limpio.Condition1)
casas_train_limpio['Condition1'] = pd.get_dummies(casas_train_limpio['Condition1'])
casas_train_limpio['HouseStyle'] = pd.Categorical(casas_train_limpio.HouseStyle)
casas_train_limpio['HouseStyle'] = pd.get_dummies(casas_train_limpio['HouseStyle'])
casas_train_limpio['ExterCond'] = pd.Categorical(casas_train_limpio.ExterCond)
casas_train_limpio['ExterCond'] = pd.get_dummies(casas_train_limpio['ExterCond'])
casas_train_limpio['Functional'] = pd.Categorical(casas_train_limpio.Functional)
casas_train_limpio['Functional'] = pd.get_dummies(casas_train_limpio['Functional'])
casas_train_limpio['PavedDrive'] = pd.Categorical(casas_train_limpio.PavedDrive)
casas_train_limpio['PavedDrive'] = pd.get_dummies(casas_train_limpio['PavedDrive'])
casas_train_limpio['SaleType'] = pd.Categorical(casas_train_limpio.SaleType)
casas_train_limpio['SaleType'] = pd.get_dummies(casas_train_limpio['SaleType'])
casas_train_limpio['SaleCondition'] = pd.Categorical(casas_train_limpio.SaleCondition)
casas_train_limpio['SaleCondition'] = pd.get_dummies(casas_train_limpio['SaleCondition'])
casas_train_limpio['BsmtCond'] = pd.Categorical(casas_train_limpio.BsmtCond)
casas_train_limpio['GarageCond'] = pd.Categorical(casas_train_limpio.GarageCond)
casas_train_limpio['BsmtCond'] = pd.get_dummies(casas_train_limpio['BsmtCond'])
casas_train_limpio['GarageCond'] = pd.get_dummies(casas_train_limpio['GarageCond'])

casas_train_limpio.BsmtCond.fillna(value='No',inplace=True)
casas_train_limpio.GarageCond.fillna(value='No',inplace=True)


data_features = casas_train_limpio.drop(['SalePrice'], axis=1).values
data_labels = casas_train_limpio['SalePrice'].values

X_train, X_test, Y_train, Y_test = train_test_split(data_features, data_labels, test_size=0.3)

#model = DecisionTreeRegressor()
model = DecisionTreeRegressor(random_state=0)
model.fit(X_train, Y_train)

score = model.score(X_test, Y_test)

print(score)
