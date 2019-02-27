'''
The objective of this script is to provide the student with the initial
tools to solve a machine learning problem
'''

# Import all necessary libraries
import pandas
import numpy
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import seaborn as sns

import plotly
import plotly.plotly as py
import plotly.graph_objs as go

import matplotlib.pyplot as plt

# Read the file, in this case a csv using pandas, and the function will return a dataframe
titanic_df = pandas.read_csv('train.csv', index_col=None, na_values=['NA'])

# ------------- Analisys phase --------------

# Dataset snapshot

# First Print the initial records to see a dataset snapshot
print(titanic_df.head(10))

# Print summary of numeric variables
print(titanic_df.describe())

# Print Data types
print(titanic_df.dtypes)

# Data missing snapshot
print(titanic_df.apply(lambda x: x.isnull().any()))

# Data missing percent
print(pandas.DataFrame({'percent_missing': titanic_df.isnull().sum() * 100 / len(titanic_df)}))

# Unique data Analisys
print(pandas.DataFrame({'percent_unique': titanic_df.apply(lambda x: x.unique().size/x.size*100)}))

plotly.tools.set_credentials_file(username='nzubiahdz', api_key='iPkUxPuEX4iQmMMRagBY')

# Class balance visualization
target_class = titanic_df.Survived.value_counts().values.tolist()

print(target_class)

data = [go.Pie(labels=['Died','Survived'], values=target_class,
              hoverinfo='label+percent', textinfo='value')]
layout = dict(
        title = "Comparison of Classes (Died/Survived)",
        autosize=False,
        width=500,
        height=500
    )

fig = dict(data=data, layout=layout)
# py.plot(fig)

def plotGraph(plot_data, msg):
	trace1 = go.Bar(
		x=plot_data.columns.values,
		y=plot_data.values[0],
		name='No'
		)
	trace2 = go.Bar(
		x=plot_data.columns.values,
		y=plot_data.values[1],
		name='Yes'
		)

	data = [trace1, trace2]

	layout = dict(
		title = msg,
		xaxis= dict(title = plot_data.columns.name),
		yaxis= dict(title= 'Number of people'),
		barmode='group',
		autosize=False,
		width=800,
		height=500
		)

	fig = dict(data=data, layout=layout)
	# py.plot(fig)

def plotLine(plotData,msg):
	trace1 = go.Scatter(
		x=plotData.columns.values,
		y=plotData.values[0],
		mode='lines',
		name='No'
		)
	trace2 = go.Scatter(
		x=plotData.columns.values,
		y=plotData.values[1],
		mode='lines',
		name='Yes'
		)

	data = [trace1, trace2]

	layout = dict(
		title = msg,
		xaxis= dict(title = plotData.columns.name),
		yaxis= dict(title= 'Number of people'),
		autosize=False,
		width=800,
		height=500
		)

	fig = dict(data=data, layout=layout)
	py.plot(fig)

# Check hypotesis women and children first
print(titanic_df.groupby(['Sex','Survived'])['Survived'].count())

sex_data = pandas.crosstab([titanic_df.Survived], titanic_df.Sex)
print(sex_data)
plotGraph(sex_data, 'Survived based on sex')

age_data = pandas.crosstab([titanic_df.Survived], titanic_df.Age)
print(age_data)
plotLine(age_data,'Survival based on Age')

# Check if high class survive more
# pandas.crosstab(titanic_df.Pclass,titanic_df.Survived,margins=True).style.background_gradient(cmap='summer_r')

p_class = pandas.crosstab([titanic_df.Survived], titanic_df.Pclass)
plotGraph(p_class,'Survived based on Pclass')


# ------------- Data preparation phase -------------------

# Drop string columns
# All values in Name and PassengerID are unique and Cabin has a lot of missing values
processed_df = titanic_df.drop(['Name', 'Cabin', 'PassengerId'],axis=1)
# processed_df = titanic_df.drop(['Name','Ticket','Cabin', 'PassengerId'],axis=1)

# Transform strings to categorical features
processed_df['Sex'] = titanic_df.Sex.astype('category')
processed_df['Embarked'] = pandas.Categorical(titanic_df.Embarked)
processed_df['Sex'] = pandas.get_dummies(processed_df['Sex'])
processed_df['Embarked'] = pandas.get_dummies(processed_df['Embarked'])

# Fill Na with values to train model
processed_df.Age.fillna(value=processed_df.Age.mean(),inplace=True)
processed_df.Embarked.fillna(value='X',inplace=True)

# Separate class from features
data_features = processed_df.drop(['Survived'], axis=1).values
data_labels = processed_df['Survived'].values

# X_train, X_test, Y_train, Y_test = train_test_split(data_features, data_labels, test_size=0.3)

# ------------ Model definition phase --------------
# clf = DecisionTreeClassifier()

# ------------ Model trainig phase -----------------
# clf.fit(X_train, Y_train)

# ------------ Model evaluation phase --------------
# score = clf.score(X_test, Y_test)

# print(score)
