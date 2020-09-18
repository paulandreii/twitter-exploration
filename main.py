import numpy as np
import pandas as pd
import json
import twitter
import requests
from requests_oauthlib import OAuth1

with open('twitter_keys.json') as f:
    keys = json.load(f)

consumer_key = keys['API Key']
consumer_secret = keys['API Secret Key']
bToken = keys['Bearer token']
access_token_key = keys['Access token']
access_token_secret = keys['Access token Secret']

api = twitter.Api(consumer_key=consumer_key,
                  consumer_secret=consumer_secret,
                  access_token_key=access_token_key,
                  access_token_secret=access_token_secret)

auth = OAuth1(consumer_key, consumer_secret,
              access_token_key, access_token_secret)


def twitter_request(req_url, auth, params):
    result = requests.get(req_url, auth=auth, params=params).json()
    return result


'''
Gets some interesting data from a list of statuses.
The list_of_statuses is a list object with dicts from statuses
key from twitter responses json.
'''


def extract_twit_data(list_of_statuses):
    twits_dataframe = pd.DataFrame()
    created_at_list = []
    id_list = []
    text_list = []
    source_list = []
    user_id_list = []
    user_location_list = []
    retweet_count_list = []
    favorite_count_list = []

    for i in range(len(list_of_statuses)):
        created_at_list.append(list_of_statuses[i]['created_at'])
        id_list.append(list_of_statuses[i]['id'])
        text_list.append(list_of_statuses[i]['text'])
        source_list.append(list_of_statuses[i]['source'])
        user_id_list.append(list_of_statuses[i]['user']['id'])
        user_location_list.append(list_of_statuses[i]['user']['location'])
        retweet_count_list.append(list_of_statuses[i]['retweet_count'])
        favorite_count_list.append(list_of_statuses[i]['favorite_count'])

    twits_dataframe['id'] = id_list
    twits_dataframe['created_at'] = created_at_list
    twits_dataframe['text'] = text_list
    twits_dataframe['source'] = source_list
    twits_dataframe['user_id'] = user_id_list
    twits_dataframe['user_location'] = user_location_list
    twits_dataframe['retweet_count'] = retweet_count_list
    twits_dataframe['favorite_count'] = favorite_count_list
    twits_dataframe.set_index('id', inplace=True)
    return twits_dataframe
