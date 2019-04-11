# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
import nltk
from nltk import sent_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Read the spam csv
spam_df = pd.read_csv('spam.csv', index_col=None, na_values=['NA'], encoding='latin1')

# print dataframe to se with wath imformation are we handling

# # print first 10 rows
# print(spam_df.head(10))
# print()

'''
Nltk
'''
# tokens = nltk.word_tokenize(text)
toktok = ToktokTokenizer()
stop_words = set(stopwords.words('english'))
# print(toktok.tokenize(spam_df.head(3)['v2']))
# print()

spam_df['v2'] = spam_df.apply(lambda spam_df: nltk.word_tokenize(spam_df['v2']), axis=1)
spam_df['v2'] = spam_df['v2'].apply(lambda x: [item for item in x if item not in stop_words])

print(spam_df.head(10))
print()

# ps = PorterStemmer()
#
# def stem_sentences(sentence):
#     tokens = sentence.split()
#     stemmed_tokens = [porter_stemmer.stem(token) for token in tokens]
#     return ' '.join(stemmed_tokens)
#
# spam_df['v2'] = spam_df['v2'].apply(stem_sentences)
#
# data['stemmed'] = data["stemmed"].apply(lambda x: [stemmer.stem(y) for y in x])


'''
vectorize df
'''
vectorize = CountVectorizer()
data_features = vectorize.fit_transform(spam_df['v2'])

# print all the words that are include on the df
# print(vectorize.vocabulary_)

# summarize encoded vector
print(data_features.shape)
print()
print(data_features.toarray())
print()


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
