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
