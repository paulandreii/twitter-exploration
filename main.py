import numpy as np
import pandas as pd
import twitter as tw

with open('twitter_keys.json') as f:
    keys = json.load(f)

key = keys['API Key']
secretkey = keys['API Secret Key']
bToken = keys['Bearer token']
