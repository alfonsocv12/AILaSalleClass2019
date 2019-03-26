import numpy as np, pandas as pd, plotly, plotly.plotly as py, plotly.graph_objs as go, matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.cluster import KMeans

rental_listing = pd.read_json('train.json')

rental_listing = rental_listing.drop(['features', 'photos'], axis=1)

# print(rental_listing.head(20))
#
# print(rental_listing.describe())
#
# print(rental_listing.dtypes)
#
# print(rental_listing.apply(lambda x: x.isnull().any()))
#
# print(pd.DataFrame({'percent_missing': rental_listing.isnull().sum() * 100 / len(rental_listing)}))
#
# print(pd.DataFrame({'percent_unique': rental_listing.apply(lambda x: x.unique().size/x.size*100)}))

rental_listing['building_id'] = pd.Categorical(rental_listing.building_id)
rental_listing['building_id'] = pd.get_dummies(rental_listing['building_id'])
rental_listing['created'] = pd.Categorical(rental_listing.created)
rental_listing['created'] = pd.get_dummies(rental_listing['created'])
rental_listing['description'] = pd.Categorical(rental_listing.description)
rental_listing['description'] = pd.get_dummies(rental_listing['description'])
rental_listing['display_address'] = pd.Categorical(rental_listing.display_address)
rental_listing['display_address'] = pd.get_dummies(rental_listing['display_address'])
# rental_listing['features'] = pd.Categorical(rental_listing.features)
# rental_listing['features'] = pd.get_dummies(rental_listing['features'])
rental_listing['manager_id'] = pd.Categorical(rental_listing.manager_id)
rental_listing['manager_id'] = pd.get_dummies(rental_listing['manager_id'])
# rental_listing['photos'] = pd.Categorical(rental_listing.photos)
# rental_listing['photos'] = pd.get_dummies(rental_listing['photos'])
rental_listing['street_address'] = pd.Categorical(rental_listing.street_address)
rental_listing['street_address'] = pd.get_dummies(rental_listing['street_address'])
rental_listing['interest_level'] = pd.Categorical(rental_listing.interest_level)
rental_listing['interest_level'] = pd.get_dummies(rental_listing['interest_level'])

'''
Plot part
'''
plotly.tools.set_credentials_file(username='alfonsocv18', api_key='IukOlHfoQOc9CejJEThc')

def plotBox(plot_data, msg):
    trace = go.Box(y=plot_data)
    data = [trace]
    layout = dict(
        title = msg
    )
    fig = dict(data=data, layout=layout)
    py.plot(fig)

def scatterPlot(x, y, msg):
    trace = go.Scatter(x = x,y = y,mode = 'markers')
    layout = dict(title=msg)
    data = [trace]
    fig = dict(data=data, layout=layout)
    py.plot(fig)

#plotBox(rental_listing['bathrooms'], 'bathrooms with outliers') #has outliers
#plotBox(rental_listing['price']) #has outliers
#plotBox(rental_listing['bedrooms'], 'bedrooms with outliers') #has outliers
#scatterPlot(rental_listing['latitude'], rental_listing['longitude']) #has outliers

''' Delete outliers '''
rental_listing_no_out = rental_listing
rental_listing_no_out = rental_listing_no_out[(rental_listing_no_out["latitude"] > 40.50) & (rental_listing_no_out["latitude"] < 40.95)]
rental_listing_no_out = rental_listing_no_out[(rental_listing_no_out["longitude"] > -74.25) & (rental_listing_no_out["longitude"] < -73.65)]
rental_listing_no_out = rental_listing_no_out[(rental_listing_no_out['bathrooms'] == 1)]
rental_listing_no_out = rental_listing_no_out[(rental_listing_no_out['bedrooms'] <=3)]
# print('')
# print(rental_listing.shape)
# print('')
# print(rental_listing_no_out.shape)
# plotBox(rental_listing_no_out['bathrooms'], 'bathrooms without outliers')
# plotBox(rental_listing_no_out['bedrooms'], 'bedrooms without outliers')

# With outliers
# data_features = rental_listing.drop(['interest_level'], axis=1).values
# data_labels = rental_listing['interest_level'].values

# without outliers
data_features = rental_listing_no_out.drop(['interest_level'], axis=1).values
data_labels = rental_listing_no_out['interest_level'].values

'''features selection'''

# test = SelectKBest(score_func=f_classif,k=5)
# fit = test.fit(data_features, data_labels)
# print('')
# print(json.dumps(dict(zip(rental_listing_no_out.columns, fit.scores_)), indent=2, sort_keys=True))
# features = fit.transform(data_features)
# print('')
# print(features[0:10])

'''clustering '''
# area = KMeans(n_clusters=5).fit_predict(rental_listing_no_out[['longitude','latitude']].values)
# print('')
# print(area[:20])

'''
Get score DecisionTreeClassifier
'''
# X_train, X_test, Y_train, Y_test = train_test_split(data_features, data_labels, test_size=0.3)
#
# model = DecisionTreeClassifier()
# model.fit(X_train, Y_train)
#
# # print(json.dumps(dict(zip(rental_listing_no_out.columns, model.feature_importances_)), indent=3, sort_keys=True))
#
# score = model.score(X_test, Y_test)
#
# print(score)
