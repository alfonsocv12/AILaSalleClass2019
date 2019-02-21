
#imports
import numpy as np, pandas as pd, plotly, plotly.plotly as py, plotly.graph_objs as go, matplotlib.pyplot as plt
# Read the file, in this case a csv using pandas, and the function will return a dataframe
casas_train = pd.read_csv('train.csv', index_col=None, na_values=['NA'])

casas_train_limpio = casas_train[['MSSubClass', 'MSZoning', 'LotArea', 'Utilities', 'Condition1',
                                  'HouseStyle', 'OverallCond','ExterCond', 'BsmtCond', '1stFlrSF',
                                  '2ndFlrSF', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Functional','GarageArea',
                                  'GarageCond','PavedDrive','MiscVal', 'SaleType','SaleCondition',
                                  'SalePrice']]
#10 valores csv
print(casas_train_limpio.head(10))
#tipos de valores
print(casas_train_limpio.dtypes)
print('')
#datos estadisticos de valores nominales
print(casas_train_limpio.describe())
print('')
#nos da el pocentaje de datos nulos
print(pd.DataFrame({'percent_missing': casas_train_limpio.isnull().sum() * 100 / len(casas_train_limpio)}))
print('')
#Nos dice que tan impotates son los datos unicos
print(pd.DataFrame({'percent_unique': casas_train_limpio.apply(lambda x: x.unique().size/x.size*100)}))

plotly.tools.set_credentials_file(username='alfonsocv18', api_key='IukOlHfoQOc9CejJEThc')

target_class = casas_train_limpio.SalePrice.value_counts().values.tolist()

def plotGraph(plot_data, msg):
    trace1 = go.Bar(x=plot_data.columns.values,y=plot_data.values[0],name='No')
    trace2 = go.Bar(x=plot_data.columns.values,y=plot_data.values[1],name='Yes')
    data = [trace1, trace2]

    layout = dict(title = msg,
    xaxis= dict(title = plot_data.columns.name),
    yaxis= dict(title= 'SalePrice'),
    barmode='group',
    autosize=False,
    width=800,
    height=500
    )
    fig = dict(data=data, layout=layout)
    py.plot(fig)

def plotLine(plotData,msg):

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
    #
    fig = dict(data=data, layout=layout)
    py.plot(fig)

print('')
#hipotesis 1
# LotArea_data = pd.crosstab(casas_train_limpio.SalePrice, casas_train_limpio.LotArea)
# #print(LotArea_data)
# plotLine(LotArea_data, 'Precio basado en LotArea')
#hipotesis 2
prueba = pd.crosstab(casas_train_limpio.SalePrice, casas_train_limpio.TotRmsAbvGrd)
plotLine(prueba, 'Sale condition')
