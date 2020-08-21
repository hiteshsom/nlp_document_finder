import os
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

import pandas as pd
import numpy as np
import random
from datetime import datetime

# Load data
import json
with open(os.path.join(os.getcwd(), 'data', 'train2.json')) as file:
    data = json.load(file)

# Preprocessing
rows = []
labels = []
for intent in data['intents']:
    rows.extend(intent['train_data'])
    label = []
    label.append(intent['label'])
    labels.extend(label*len(intent['train_data']))


# Date recognition
from datetime import datetime

def decode_date(tokens):

    months_fullform = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'decemeber']
    months_shortform = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    month = 0
    for token in tokens:
        try:
            int(token)
        except:
            if token.lower() in months_fullform:
                month = months_fullform.index(token.lower()) + 1
            elif token.lower() in months_shortform:
                month = months_shortform.index(token.lower()) + 1
            tokens.remove(token)
        
    if len(tokens) > 2:
        if month:
            day = int(tokens[0])
            year = int(tokens[2])
        else:
            day = int(tokens[0])
            month = int(tokens[1])
            year = int(tokens[2])
    elif len(tokens) > 1:
        if month:
            day = int(tokens[0])
            year = 2020
        else:
            day = int(tokens[0])
            month = int(tokens[1])
            year = 2020
    elif len(tokens)==1:
        day = int(tokens[0])
        year = 2020
    return (day, month, year)


# !python3 -m spacy download en_core_web_lg
import spacy
sp_lg = spacy.load('en_core_web_lg')
import nltk
import re

query_date = {}
for idx, row in enumerate(rows):
    date_occurences = [(ent.text.strip(), ent.label_) for ent in sp_lg(row).ents if ent.label_ == 'DATE']
    query_date[row] = []
    
    for date in date_occurences:
        try:
            date_token = re.split('\s+|/|-|:',date[0])
            day, month, year = decode_date(date_token)
            query_date[row].append((f'{year}-{month}-{day}'))
            row = row.replace(date[0], "", 1)
            rows[idx] = row
        except:
            pass


# Universal Sentence Encoder
import tensorflow_hub as hub
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

embedding = embed(rows)

X = pd.DataFrame(data=embedding.numpy())
X['label'] = labels

import seaborn as sns
import matplotlib.pyplot as plt

def plot_similarity(labels, features, rotation):
    corr = np.inner(features, features)
    sns.set(font_scale=1)
    fig, ax = plt.subplots(figsize=(18, 18))
    g = sns.heatmap(
      corr,
      xticklabels=labels,
      yticklabels=labels,
      vmin=0,
      vmax=1,
      cmap="YlOrRd", 
      ax=ax)
    g.set_xticklabels(labels, rotation=rotation)
    g.set_title("Semantic Textual Similarity")

def run_and_plot(messages_):
    message_embeddings_ = embed(messages_)
    plot_similarity(messages_, message_embeddings_, 90)


messages = rows

# run_and_plot(messages)

#t-SNE
from sklearn.manifold import TSNE
import time

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=10, n_iter=300)
tsne_results = tsne.fit_transform(X.drop(columns=['label']))
# print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

import matplotlib.pyplot as plt
import seaborn as sns

# fig, ax = plt.subplots(figsize=(12,9))
# ax.scatter(tsne_results[0:18, 0], tsne_results[0:18, 1])
# plt.show()

df_subset = pd.DataFrame()
df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]
df_subset['label'] = X['label']

# plt.figure(figsize=(26,16))
# sns.scatterplot(
#     x="tsne-2d-one", 
#     y="tsne-2d-two",
#     hue="label",
#     palette=sns.color_palette("hls", 14),
#     data=df_subset,
#     legend="full",
#     alpha=1,
#     s=100
# )


# Modelling
# Data Prepraration
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

y = X['label']
X = X.drop(columns=['label'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=10)
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Building Neural network
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import datetime

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model = Sequential()
model.add(Dense(64, input_shape=(512,)))
model.add(Dense(100))
model.add(Dense(1000))
model.add(Dense(100))
model.add(Dense(len(set(y_train))))
model.add(Activation('softmax'))

model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_test, y_test))

y_proba = model.predict(X_test)

y_pred = y_proba.argmax(axis=1)

from operator import itemgetter
final_results = pd.DataFrame(data = list(itemgetter(*list(X_test.index))(rows)), index=X_test.index, columns=['inputs'])  

final_results['predictions'] = le.inverse_transform(y_pred)
final_results['true'] = le.inverse_transform(y_test)
final_results['query_search'] = ""

for index in final_results.index:
    for intent in data['intents']:
        if intent['label'] == final_results.loc[index, 'predictions']:
            final_results.loc[index, 'query_search'] = intent['responses']
            # print(intent['responses'])


import joblib

joblib.dump(le, 'le.joblib')

model.save('model')
                
