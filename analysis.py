import numpy as np
import tweepy as tw
import random, nltk
from gensim.models import Word2Vec
from keras.models import model_from_json

def get_word2vec_model():
    return Word2Vec.load('./models/word2vec/word2vec.model')

def get_lstm_model():
    json_file = open('./models/lstm/lstm.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights('./models/lstm/lstm.h5') 
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

def get_lstm_data(tweets):
    wordvecModel = get_word2vec_model()
    X = []
    for tweet in tweets:
        tokens = nltk.word_tokenize(tweet)
        embeddings = []
        for token in tokens:
            try:
                embeddings.append([round(abs(sum(wordvecModel[token])) * 10, 4)])
            except:
                embeddings.append([round(10 * random.random(), 4)])
        padding = [[0]] * (128 - len(embeddings))
        embeddings = embeddings.copy() + padding
        X.append(embeddings)
    X = np.array(X)
    return X

def connect():
    consumer_key = 'jPRTv2TEp4DnffcAsH0Y5YT2U'
    consumer_secret = 'Lmfb7x6sPXQ4rkCjQ1kzABr8KsRZdEJb0JDxFWUcNnhopGCsq0'
    access_token = '148014695-CARCJ7B2y9yo5Ap7cNxkJqENBAavZHEbDQQ2oF27'
    access_token_secret = 'nvYlgWqC7bOEMUDs0o9C42u0mJpCegvKDE8v9jOCo6oOg'
    auth = tw.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tw.API(auth, wait_on_rate_limit=True)
    return api

def depression_analysis(handle):
    api = connect()
    tweets = api.user_timeline(
        screen_name=handle, 
        count=100,
        include_rts=False,
        tweet_mode='extended'
    )
    tweets_text = []
    for info in tweets:
        tweets_text.append(info.full_text)
    X = get_lstm_data(tweets_text)
    model = get_lstm_model()
    pred_output = model.predict(X)
    sum = 0
    for output in pred_output:
        if output[0] > output[1]:
            sum += output[0]
        
        else:
            sum += output[1]
            
    return { 'index': round(sum / len(pred_output) * 100) }
    
    