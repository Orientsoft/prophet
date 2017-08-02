import multiprocessing
from pymongo import MongoClient

class TimestampConvertor(object):
    def __init__(self, db, collection, mongo_url='mongodb://localhost:27017'):
        self.db = db
        self.collection = collection

        self.mongo_url = mongo_url
        self.client = MongoClient(self.mongo_url)
