import numpy as np
import pandas as pd
import json
import twitter
import tweepy
import requests
from requests_oauthlib import OAuth1


class DataExtraction:
    '''
    Create a DataExtraction objet used to get data from the twitter api
    '''

    def __init__(self, keys_json_path):
        '''
        Initialize object. It stores info about your twitter API auth.
        Create and api object with all the methods from the twitter python
        library.
        '''
        self.keys_json_path = keys_json_path
        with open(self.keys_json_path) as f:
            self.keys = json.load(f)
        self.consumer_key = self.keys['API Key']
        self.consumer_secret = self.keys['API Secret Key']
        self.bToken = self.keys['Bearer token']
        self.access_token_key = self.keys['Access token']
        self.access_token_secret = self.keys['Access token Secret']

        self.auth = OAuth1(self.consumer_key, self.consumer_secret,
                           self.access_token_key, self.access_token_secret)

        self.api = twitter.Api(consumer_key=self.consumer_key,
                               consumer_secret=self.consumer_secret,
                               access_token_key=self.access_token_key,
                               access_token_secret=self.access_token_secret)
        self.auth_tweepy = tweepy.OAuthHandler(
            self.consumer_key, self.consumer_secret)
        self.auth_tweepy.set_access_token(
            self.access_token_key, self.access_token_secret)
        self.tweepy_api = tweepy.API(self.auth_tweepy, wait_on_rate_limit=True)

    def twitter_request(self, req_url, params):
        '''
        Takes an API url and its parameters to return a twitter API
        json object.
        '''
        result = requests.get(req_url, auth=self.auth, params=params).json()
        return result

    def extract_twit_data(self, list_of_statuses):
        '''
        Takes a list with all the twits taken from the json response
        from the twitter API search method.
        Outputs a dataframe with some interesting info about the twits.
        '''
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

    def get_all_hashtag_data(self, hashtag_name, lang="es", since="2020-07-21"):
        '''
        Gets all the tweets for the specified hashtag, language and since that date.
        '''
        list_of_tweets = []
        for tweet in tweepy.Cursor(self.tweepy_api.search, q=hashtag_name, count=100,
                                   lang=lang,
                                   since=since).items():
            print(tweet.text)
            list_of_tweets.append(tweet)
        return list_of_tweets
