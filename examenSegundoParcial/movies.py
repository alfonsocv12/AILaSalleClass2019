import numpy as np, pandas as pd, json, ast
from pandas.io.json import json_normalize
from collections import Counter
from collections import Counter
from controllers.plot_controller import plotController as pc
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report

movies = pd.read_csv('train.csv',index_col=None, na_values=['NA'])

# json_normalize(data=movies, record_path=movies, meta=['belongs_to_collection'])


#10 valores csv
# print(movies.head(10))
#tipos de valores
# print(movies.dtypes)
# print('')
# #datos estadisticos de valores nominales
# print(movies.describe())
# print('')
# #nos da el pocentaje de datos nulos
# print(pd.DataFrame({'percent_missing': movies.isnull().sum() * 100 / len(movies)}))
# print('')
# #Nos dice que tan impotates son los datos unicos
# print(pd.DataFrame({'percent_unique': movies.apply(lambda x: x.unique().size/x.size*100)}))

movies = movies.drop(['poster_path','id','imdb_id'], axis=1)
#
# print(movies.belongs_to_collection)
#
# for movie in movies:
#     print(movie)
#
# print(movies.head(1).belongs_to_collection.to_string())

# pd.set_option('display.max_colwidth', -1)
# print(movies['belongs_to_collection'].head(1))

''' transform json NaN to {} '''


# print before transform
# for i, e in enumerate(movies['release_date'][:5]):
#     print(i, e)

dict_columns = ['belongs_to_collection', 'genres', 'production_companies',
                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']

def text_to_dict(df):
    for column in dict_columns:
        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x) )
    return df

movies = text_to_dict(movies)

# print after transform
# for i, e in enumerate(movies['spoken_languages'][:5]):
#     print(i, e)

''' Experimenting with genres '''
# print('')
# print(movies['genres'].apply(lambda x: len(x) if x != {} else 0).value_counts())

list_of_genres = list(movies['genres'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)

# print('')
# print(list_of_genres)
# print('')
# print(Counter([i for j in list_of_genres for i in j]).most_common())

''' Experimenting with production_companies, production_countries, spoken_languages'''

# print(movies['production_companies'].apply(lambda x: len(x) if x != {} else 0).value_counts())
list_of_companies = list(movies['production_companies'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
list_of_countries = list(movies['production_countries'].apply(lambda x: [i['iso_3166_1'] for i in x] if x != {} else []).values)
list_of_spoken_languages = list(movies['spoken_languages'].apply(lambda x: [i['iso_639_1'] for i in x] if x != {} else []).values)
# print(Counter([i for j in list_of_companies for i in j]).most_common(30))

''' Experimenting with Keywords'''

# for i, e in enumerate(movies['Keywords'][:5]):
#     print(i, e)

# print(movies['Keywords'].apply(lambda x: len(x) if x != {} else 0).value_counts())
list_of_keywords = list(movies['Keywords'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
# print(Counter([i for j in list_of_keywords for i in j]).most_common(50))

'''Experimenting with cast'''
# for i, e in enumerate(movies['cast'][:1]):
#     print(i, e)
list_of_cast_names = list(movies['cast'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
# print(Counter([i for j in list_of_cast_names for i in j]).most_common(15))

list_of_cast_genders = list(movies['cast'].apply(lambda x: [i['gender'] for i in x] if x != {} else []).values)
# print(Counter([i for j in list_of_cast_genders for i in j]).most_common())

list_of_cast_characters = list(movies['cast'].apply(lambda x: [i['character'] for i in x] if x != {} else []).values)
# print(Counter([i for j in list_of_cast_characters for i in j]).most_common(15))

'''Experimenting with the crew'''

# for i, e in enumerate(movies['crew'][:1]):
#     print(i, e[:10])

# print(movies['crew'].apply(lambda x: len(x) if x != {} else 0).value_counts().head(10))
# print('name')
list_of_crew_names = list(movies['crew'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
# print(Counter([i for j in list_of_crew_names for i in j]).most_common(15))
# print('job')
list_of_crew_jobs = list(movies['crew'].apply(lambda x: [i['job'] for i in x] if x != {} else []).values)
# print(Counter([i for j in list_of_crew_jobs for i in j]).most_common(15))
# print('gender')
list_of_crew_genders = list(movies['crew'].apply(lambda x: [i['gender'] for i in x] if x != {} else []).values)
# print(Counter([i for j in list_of_crew_genders for i in j]).most_common(15))
# print('department')
list_of_crew_departments = list(movies['crew'].apply(lambda x: [i['department'] for i in x] if x != {} else []).values)
# print(Counter([i for j in list_of_crew_departments for i in j]).most_common(14))


''' transform json to columns'''
# print after transform

# belongs_to_collection
movies['has_collection'] = movies['belongs_to_collection'].apply(lambda x: len(x) if x != {} else 0)
# movies['belongs_to_collection_name']= movies['belongs_to_collection'].apply(lambda x: x[0]['name'] if x != {} else None)

#Genres
movies['num_genres'] = movies['genres'].apply(lambda x: len(x) if x != {} else 0)

all_genres = movies['genres'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')

top_genres = [m[0] for m in Counter([i for j in list_of_genres for i in j]).most_common(5)]
for g in top_genres:
    movies['genre_' + g] = all_genres.apply(lambda x: 1 if g in x else 0)

#Companies

movies['num_companies'] = movies['production_companies'].apply(lambda x: len(x) if x != {} else 0)
# movies['all_production_companies'] = movies['production_companies'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
# top_companies = [m[0] for m in Counter([i for j in list_of_companies for i in j]).most_common(30)]
# for g in top_companies:
#     movies['production_company_' + g] = movies['all_production_companies'].apply(lambda x: 1 if g in x else None)

#Countries
movies['num_countries'] = movies['production_countries'].apply(lambda x: len(x) if x != {} else 0)

all_countries = movies['production_countries'].apply(lambda x: ' '.join(sorted([i['iso_3166_1'] for i in x])) if x != {} else '')

# top_countries = [m[0] for m in Counter([i for j in list_of_countries for i in j]).most_common(5)]
# for g in top_countries:
#     movies['production_country_' + g] = all_countries.apply(lambda x: 1 if g in x else 0)

#spoken_languages
movies['num_languages'] = movies['spoken_languages'].apply(lambda x: len(x) if x != {} and len(x) > 1 else 0)
all_languages = movies['spoken_languages'].apply(lambda x: ' '.join(sorted([i['iso_639_1'] for i in x])) if x != {} else '')
top_languages = [m[0] for m in Counter([i for j in list_of_spoken_languages for i in j]).most_common(4)]
for g in top_languages:
    movies['language_' + g] = all_languages.apply(lambda x: 1 if g in x else 0)

#Keywords
movies['num_Keywords'] = movies['Keywords'].apply(lambda x: len(x) if x != {} else 0)
all_Keywords = movies['Keywords'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
# top_keywords = [m[0] for m in Counter([i for j in list_of_keywords for i in j]).most_common(5)]
# for g in top_keywords:
#     movies['keyword_' + g] = all_Keywords.apply(lambda x: 1 if g in x else 0)

# Cast
movies['num_cast'] = movies['cast'].apply(lambda x: len(x) if x != {} else 0)
top_cast_names = [m[0] for m in Counter([i for j in list_of_cast_names for i in j]).most_common(15)]
# for g in top_cast_names:
#     movies['cast_name_' + g] = movies['cast'].apply(lambda x: 1 if g in str(x) else 0)
movies['genders_0_cast'] = movies['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]) if x != 0 else 0)
movies['genders_1_cast'] = movies['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]) if x != 0 else 0)
movies['genders_2_cast'] = movies['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]) if x != 0 else 0)
# top_cast_characters = [m[0] for m in Counter([i for j in list_of_cast_characters for i in j]).most_common(15)]
# for g in top_cast_characters:
#     movies['cast_character_' + g] = movies['cast'].apply(lambda x: 1 if g in str(x) else 0)

#crew

# movies['num_crew'] = movies['crew'].apply(lambda x: len(x) if x != {} else 0)

top_crew_names = [m[0] for m in Counter([i for j in list_of_crew_names for i in j]).most_common(15)]
# for g in top_crew_names:
#     movies['crew_name_' + g] = movies['crew'].apply(lambda x: 1 if g in str(x) else 0)

movies['genders_0_crew'] = movies['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]) if x != 0 else 0)
movies['genders_1_crew'] = movies['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]) if x != 0 else 0)
movies['genders_2_crew'] = movies['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]) if x != 0 else 0)

top_crew_jobs = [m[0] for m in Counter([i for j in list_of_crew_jobs for i in j]).most_common(15)]
for j in top_crew_jobs:
    movies['jobs_' + j] = movies['crew'].apply(lambda x: sum([1 for i in x if i['job'] == j]) if x != 0 else 0)

top_crew_departments = [m[0] for m in Counter([i for j in list_of_crew_departments for i in j]).most_common(15)]
for j in top_crew_departments:
    movies['departments_' + j] = movies['crew'].apply(lambda x: sum([1 for i in x if i['department'] == j]) if x != 0 else 0)



'''removing outliers, get_dummies'''


# Geting ready the data
# movies['homepage'] = pd.Categorical(movies.homepage)
# movies['homepage'] = pd.get_dummies(movies['homepage'])
movies['original_language'] = pd.Categorical(movies.original_language)
movies['original_language'] = pd.get_dummies(movies['original_language'])
movies['popularity'] = movies['popularity'].round(0)
def fix_date(x):
    """
    Fixes dates which are in 20xx
    """
    year = x.split('/')[2]
    if int(year) <= 19:
        return int('20' + year)
    else:
        return int('19' + year)

movies['release_date'] = movies['release_date'].apply(lambda x: fix_date(x))
movies.runtime.fillna(value='90.0', inplace=True)
# movies['pa'] = pd.Categorical(movies.pa)
# movies['pa'] = pd.get_dummies(movies['pa'])
# movies['pa'] = pd.Categorical(movies.pa)
# movies['pa'] = pd.get_dummies(movies['pa'])
# movies['pa'] = pd.Categorical(movies.pa)
# movies['pa'] = pd.get_dummies(movies['pa'])

# removing outliers
movies = movies[(movies['num_genres'] > 0) & (movies['num_genres'] < 7 )]
movies = movies[(movies['num_companies'] < 7)]
movies = movies[(movies['num_countries'] < 2)]
movies = movies[(movies['popularity'] < 21)]
movies = movies[(movies['release_date'] > 1959)]
movies = movies[(movies['budget']< 6200000)]

# Print transforming data
# print(Counter(movies['runtime'].values).most_common())
# for i, e in enumerate(movies['has_collection'][:5]):
#     print(i, e)
# print(movies['production_country_US'].value_counts())

#Showing results in plots
# pc.plotBox(movies['budget'], 'budget')
# pc.plotBox(movies['release_date'], 'release_date')
# pc.plotBox(movies['popularity'], 'popularity')
# pc.plotBox(movies['num_companies'], 'num_companies')
# pc.plotBox(movies['num_countries'], 'num_countries')
# pc.scatterPlot(movies['num_cast'], movies['revenue'], 'num_cast')

''' removing none use classes '''

movies = movies.drop(['belongs_to_collection', 'genres', 'production_companies',
                      'production_countries', 'spoken_languages', 'Keywords',
                      'cast', 'crew', 'overview',
                      'original_title', 'title', 'status',
                      'homepage', 'tagline'
                      ], axis=1)

''' Getting features and labels'''

data_features = movies.drop(['revenue'], axis=1).values
data_labels = movies['revenue'].values


''' Smothing and dimension reduction'''

test = SelectKBest(score_func=f_classif,k=10)
fit = test.fit(data_features, data_labels)
# print('')
# print(json.dumps(dict(zip(movies.columns, fit.scores_)), indent=2, sort_keys=True))
features = fit.transform(data_features)

scaler = MinMaxScaler(feature_range=(0,1))
features = scaler.fit_transform(data_features)

'''Experimenting with data out ofter new colums add '''

pd.set_option('display.max_rows', len(movies))
pd.set_option('display.max_columns', len(movies))


# #10 valores csv
# print(movies.head(1))
# # tipos de valores
# print(movies.dtypes)
# print('')
# #datos estadisticos de valores nominales
# print(movies.describe())
# print('')
# print(pd.DataFrame({'percent_missing': movies.isnull().sum() * 100 / len(movies)}))
# print('')
# print(pd.DataFrame({'percent_unique': movies.apply(lambda x: x.unique().size/x.size*100)}))

'''Applying models '''

X_train, X_test, Y_train, Y_test = train_test_split(features, data_labels, test_size=0.3)

model = DecisionTreeRegressor(max_depth=7)
model.fit(X_train, Y_train)

names = ['Decision Tree regressors', 'MLP regressors',
         'Random Forest regressors', 'AdaBoost',
         'Bagging regressors']

regressors = [
  DecisionTreeRegressor(max_depth=7),
  MLPRegressor(alpha=1),
  RandomForestRegressor(max_depth=5, max_features=10, n_estimators=10),
  AdaBoostRegressor(n_estimators=10),
  BaggingRegressor(max_features=1,n_estimators=10)
]

for name, regressor in zip(names, regressors):
    regressor.fit(X_train,Y_train)
    y_pred = regressor.predict(X_test)
    print('{}: {}'.format(name, mean_squared_error(Y_test, y_pred)))

# score = mean_squared_error(Y_test, model.predict(X_test))
#
# print(score)
