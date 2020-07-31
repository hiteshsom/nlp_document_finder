from flask import Flask, render_template, request
import spacy
import pandas as pd 
import numpy as np
sp_lg = spacy.load('en_core_web_lg')
import nltk
import re
import json
with open("data/train2.json") as file:
    data = json.load(file)
import tensorflow_hub as hub
import tensorflow as tf
import re
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

import joblib

le = joblib.load('le.joblib')


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



app = Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            original_user_input = request.form['user_input']
            user_input = [original_user_input]
            query_date = {}
            for idx, row in enumerate(user_input):
                date_occurences = [(ent.text.strip(), ent.label_) for ent in sp_lg(row).ents if ent.label_ == 'DATE']
                query_date[row] = []
                
                for date in date_occurences:
                    try:
                        date_token = re.split('\s+|/|-|:',date[0])
                        day, month, year = decode_date(date_token)
                        query_date[row].append((day, month, year))
                        row = row.replace(date[0], "", 1)
                        user_input[idx] = row
                    except:
                        pass
            embedding = embed(user_input)
            X_user = pd.DataFrame(data=embedding.numpy())

            model = tf.keras.models.load_model('model')
            y_proba = model.predict(X_user)
            y_pred = y_proba.argmax(axis=1)
            y_proba = model.predict(X_user)
            y_pred = y_proba.argmax(axis=1)
            query_document = []
            for idx, pred in enumerate(y_pred):
                for intent in data['intents']:
                    if intent['label'] == le.inverse_transform(pred.reshape(1,)):
                        query_document.append(intent['responses'])

        except ValueError:
            return "Check if text is entered correctly"
    
    return render_template('predict.html', prediction = list(zip([original_user_input], query_document, query_date.values()))) 


if __name__ == "__main__":
    app.run(port=10000, debug=True)