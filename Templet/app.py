# Flask is the overall web framework
from flask import Flask, request, jsonify, render_template
import json

# joblib is used to unpickle the model

import joblib
import pandas as pd
import numpy as np
import flask
import pickle
import pandas as pd
import numpy as np
import nltk
import string
import re


# Use pickle to load in the pre-trained model.
with open(f'model/logistic.pkl', 'rb') as f:
    model = pickle.load(f)


with open(f'model/vectorize.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


def remove_punctuation(words):
    """Remove punctuation from a list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

#Tokenizer
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')



#Removing Stop words
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def remove_stopwords(Summary):
    # Use list comprehension for efficient list creation
    new_Summary = [word for word in Summary.split() if word not in stop_words]
    return " ".join(new_Summary)


#Lemmatizaion
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def to_lowercase(words):
    """Convert all characters to lowercase from a list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words



app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])

def index():
    
    if flask.request.method == 'GET':
        return(flask.render_template('index.html'))
    
    if flask.request.method == 'POST':
        
        Summary = flask.request.form['']

        df = pd.DataFrame([Summary], columns=['Summary'])



        
        df['Summary'] = df['Summary'].apply(lambda word: remove_punctuation(word))

        df['Summary'] = df['Summary'].apply(lambda word: tokenizer.tokenize(t.lower()))

        df['Summary'] = df['Summary'].apply(lambda t: remove_sw(t))

        df['Summary'] = df['Summary'].apply(lambda t: word_lemmatizer(t))




        final_text = df['tweet']

        final_text.iloc[0] = ' '.join(final_text.iloc[0])

        final_text = vectorizer.transform(final_text)



        prediction = model.predict(final_text)
        
        return flask.render_template('index.html', result=prediction, original_input={'Mobile Review':tweet})




if __name__ == '__main__':
    app.run()