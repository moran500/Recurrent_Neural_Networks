import pandas as pd
import pymongo
import json

mng_client = pymongo.MongoClient('localhost', 27017)
mng_db = mng_client['local'] 
collection_name = 'stock_prices'
db_cm = mng_db[collection_name]
data = pd.read_csv('Google_Stock_Prices_Total.csv')
data_json = json.loads(data.to_json(orient='records'))
db_cm.insert(data_json)

