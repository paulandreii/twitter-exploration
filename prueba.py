import data_extraction as de
import pandas as pd
import json
import tweepy
from requests_oauthlib import OAuth1

pd.set_option('display.max_columns', 5000)
pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_colwidth', 5000)

with open('twitter_keys.json') as f:
    keys = json.load(f)

consumer_key = keys['API Key']
consumer_secret = keys['API Secret Key']
bToken = keys['Bearer token']
access_token_key = keys['Access token']
access_token_secret = keys['Access token Secret']

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token_key, access_token_secret)

keys_json_path = 'twitter_keys.json'

data_extractor = de.DataExtraction(keys_json_path)

list_of_tweets = data_extractor.get_all_hashtag_data(
    hashtag_name="M-30", lang="es", since="2020-07-27")

m30data = pd.read_csv('M-30.csv', encoding='utf-8')

m30data.set_index('id', drop=True, inplace=True)

print(m30data)
