import numpy as np, pandas as pd, plotly, plotly.plotly as py, plotly.graph_objs as go, matplotlib.pyplot as plt, cufflinks as cf

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

rental_listing = pd.read_json('train.json')

rental_listing = rental_listing.drop(['features', 'photos'], axis=1)

print(rental_listing.head(20))

print(rental_listing.describe())

print(rental_listing.dtypes)

print(rental_listing.apply(lambda x: x.isnull().any()))

print(pd.DataFrame({'percent_missing': rental_listing.isnull().sum() * 100 / len(rental_listing)}))

print(pd.DataFrame({'percent_unique': rental_listing.apply(lambda x: x.unique().size/x.size*100)}))

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

plotly.tools.set_credentials_file(username='alfonsocv18', api_key='IukOlHfoQOc9CejJEThc')

def plotBox(plot_data, msg):
    cf.set_config_file(offline=False, world_readable=True, theme='ggplot')
    plot_data.iplot(kind='box',filename=msg)
    # trace0 = go.Scatter(x = plot_data.columns, y = plot_data.mean(), mode='lines', name='mean')
    # layout = go.Layout(
    #     yaxis=dict(
    #         title=msg,
    #         zeroline=False
    #     ),
    #     boxmode='group'
    # )
    # data = [trace0]
    # fig = go.Figure(data=data, layout=layout)





rental_listing_2 = rental_listing.head(20)
data_labels = rental_listing_2['interest_level'].values
bathrooms_dataframe = pd.DataFrame(index=rental_listing_2.bathrooms, columns=data_labels)
print(bathrooms_dataframe)
# plotBox(bathrooms_dataframe, 'Interes en base a la cantidad de banos')

# X_train, X_test, Y_train, Y_test = train_test_split(data_features, data_labels, test_size=0.3)

# model = DecisionTreeClassifier()
# model.fit(X_train, Y_train)
#
# score = model.score(X_test, Y_test)
#
# print(score)
