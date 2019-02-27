import numpy as np, pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

rental_listing = pd.read_json('train.json')

rental_listing = rental_listing.drop(['features', 'photos'], axis=1)

print(rental_listing.head(10))

print(rental_listing.describe())

print(rental_listing.dtypes)

print(rental_listing.apply(lambda x: x.isnull().any()))

#print(pd.DataFrame({'percent_missing': rental_listing.isnull().sum() * 100 / len(rental_listing)}))

#print(pd.DataFrame({'percent_unique': rental_listing.apply(lambda x: x.unique().size/x.size*100)}))

# target_class = rental_listing.interest_level.value_counts().values.tolist()
#
# print(target_class)

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

data_features = rental_listing.values
data_labels = rental_listing['interest_level'].values

X_train, X_test, Y_train, Y_test = train_test_split(data_features, data_labels, test_size=0.3)

model = DecisionTreeClassifier()
model.fit(X_train, Y_train)

score = model.score(X_test, Y_test)

print(score)
