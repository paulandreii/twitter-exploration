import numpy as np
import pandas as pd
import json
import twitter
import tweepy
import requests
from requests_oauthlib import OAuth1
import csv_writer
import psycopg2
from config import config
import datetime as dt


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
        user_name_list = []
        user_location_list = []
        retweet_count_list = []
        favorite_count_list = []

        for i in range(len(list_of_statuses)):
            created_at_list.append(list_of_statuses[i]['created_at'])
            id_list.append(list_of_statuses[i]['id'])
            # needs tweet_mode extended in params when called
            text_list.append(list_of_statuses[i]['full_text'])
            source_list.append(list_of_statuses[i]['source'])
            user_id_list.append(list_of_statuses[i]['user']['id'])
            user_name_list.append(list_of_statuses[i]['user']['screen_name'])
            user_location_list.append(list_of_statuses[i]['user']['location'])
            retweet_count_list.append(list_of_statuses[i]['retweet_count'])
            favorite_count_list.append(list_of_statuses[i]['favorite_count'])

        twits_dataframe['id'] = id_list
        twits_dataframe['created_at'] = created_at_list
        twits_dataframe['full_text'] = text_list
        twits_dataframe['source'] = source_list
        twits_dataframe['user_id'] = user_id_list
        twits_dataframe['user_name'] = user_name_list
        twits_dataframe['user_location'] = user_location_list
        twits_dataframe['retweet_count'] = retweet_count_list
        twits_dataframe['favorite_count'] = favorite_count_list
        twits_dataframe.set_index('id', inplace=True)
        return twits_dataframe

    def get_all_hashtag_data_csv(self, hashtag_name, lang="es", since="2020-07-21"):
        '''
        Gets all the tweets for the specified hashtag, language and since that date.
        '''
        list_of_tweets = []
        tweetWriter = csv_writer.TweetCSVWriter(csv_name=hashtag_name, column_names=[
            'id', 'created_at', 'full_text', 'source', 'user_id', 'user_location', 'retweet_count', 'favorite_count'], separator=",")
        for tweet in tweepy.Cursor(self.tweepy_api.search, q=hashtag_name, count=100,
                                   lang=lang,
                                   since=since,
                                   tweet_mode='extended').items():
            tweet_data = [tweet.id, tweet.created_at, tweet.full_text.encode('utf-8'), tweet.source,
                          tweet.user.id, tweet.user.location.encode('utf-8'), tweet.retweet_count, tweet.favorite_count]

            # print(tweet_data)

            tweetWriter.tweet_writer(tweet_data)
            list_of_tweets.append(tweet_data)
            if len(list_of_tweets) == 3000:
                break
        return list_of_tweets

    def get_all_hashtag_data_postgres(self, hashtag_name, lang="es", since="2020-07-21"):
        '''
        Gets all the tweets for the specified hashtag, language and since that date.
        '''
        f = '%Y-%m-%d %H:%M:%S%Z'
        i = 0

        db_params = config()
        conn = psycopg2.connect(**db_params)
        db_cursor = conn.cursor()

        table_name = hashtag_name.replace(" ", "_")

        try:
            db_cursor.execute('CREATE TABLE ' + table_name +
                              ' (id BIGINT, created_at timestamp null, full_text text null, source text null, user_id BIGINT null, user_location text null, retweet_count int null, favorite_count int null)')
        except Exception as e:
            print(e)

        for tweet in tweepy.Cursor(self.tweepy_api.search, q=hashtag_name, count=100,
                                   lang=lang,
                                   since=since,
                                   tweet_mode='extended',
                                   exclude='retweets').items():
            tweet_data = {'id': tweet.id, 'created_at': tweet.created_at.strftime(f), 'full_text': tweet.full_text, 'source': tweet.source,
                          'user_id': tweet.user.id, 'user_location': tweet.user.location, 'retweet_count': tweet.retweet_count, 'favorite_count': tweet.favorite_count}

            try:
                db_cursor.execute('INSERT INTO ' + table_name +
                                  ' (id, created_at, full_text, source, user_id, user_location, retweet_count, favorite_count) VALUES (%(id)s, %(created_at)s, %(full_text)s, %(source)s, %(user_id)s, %(user_location)s, %(retweet_count)s, %(favorite_count)s);', tweet_data)
            except Exception as e:
                print(e)

            i += 1
            if i == 60000:
                break
        conn.commit()
        db_cursor.close()
        conn.close()
        return

    def tweet_xlsx_file(self, df):
        '''
        Writes an XLSX files from a specific twits_dataframe
        '''
        print('Write XLSX file name')
        xlsxfilename = input()
        writer = pd.ExcelWriter(xlsxfilename+".xlsx", engine="xlsxwriter")
        df.to_excel(writer, sheet_name=xlsxfilename)
        workbook = writer.book
        worksheet = writer.sheets[xlsxfilename]
        # Set the columns width and format
        # ID width and format
        formatid = workbook.add_format({'num_format': '0'})
        worksheet.set_column(0, 0, 30, formatid)
        # Created At width
        worksheet.set_column(1, 1, 35)
        # Full_text width (gonna be long)
        worksheet.set_column(2, 2, 280)
        # source width
        worksheet.set_column(3, 3, 80)
        # user_id width and format
        formatuid = workbook.add_format({'num_format': '0'})
        worksheet.set_column(4, 4, 20, formatuid)
        # user_location width
        worksheet.set_column(5, 5, 30)
        # retweet_width
        worksheet.set_column(6, 6, 15)
        # favorite_count width
        worksheet.set_column(7, 7, 15)
        writer.save()
