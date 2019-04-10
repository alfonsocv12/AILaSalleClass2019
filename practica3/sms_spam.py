# -*- coding: utf-8 -*-
import pandas as pd, numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Read the spam csv
spam_df = pd.read_csv('spam.csv', index_col=None, na_values=['NA'], encoding='latin1')

# print dataframe to se with wath imformation are we handling

# # print first 10 rows
# print(spam_df.head(10))
# print()
#
# # print describe info abouth the columns
# print(spam_df.describe)
# print()
#
# # Print Data types
# print(spam_df.dtypes)
# print()
#
# # Data missing snapshot
# print(spam_df.apply(lambda x: x.isnull().any()))
# print()
#
# # Data missing percent
# print(pd.DataFrame({'percent_missing': spam_df.isnull().sum() * 100 / len(spam_df)}))
# print()
#
# # Unique data Analisys
# print(pd.DataFrame({'percent_unique': spam_df.apply(lambda x: x.unique().size/x.size*100)}))
# print()

'''
vectorize df
'''
vectorize = CountVectorizer()
data_features = vectorize.fit_transform(spam_df['v2'])

# print all the words that are include on the df
# print(vectorize.vocabulary_)

# # summarize encoded vector
# print(data_features.shape)
# print()
# print(data_features.toarray())
# print()

'''
Train test split
'''
data_labels = spam_df['v1'].values

X_train, X_test, Y_train, Y_test = train_test_split(data_features, data_labels, test_size=0.3)

'''
Using MultinomialNB model
'''

clf = MultinomialNB()
clf.fit(X_train, Y_train)

score = clf.score(X_test, Y_test)
print(score)
