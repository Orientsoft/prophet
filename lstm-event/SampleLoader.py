# -*- coding:utf-8 -*-

import sys
sys.path.append("../data-preprocessing")
import WindowNormalizer

import multiprocessing
import numpy as np
import pymongo
from pymongo import MongoClient

class SampleLoader(object):
    def __init__(self, db_name, collection_prefix, mongo_url="mongodb://localhost:27017", batch_size=4, data_length=5, label_length=1, random_seed=int(9527), queue_size=32):
        self.db_name = db_name
        self.collection_prefix = collection_prefix
        self.mongo_url = mongo_url

        self.batch_size = batch_size
        self.data_length = data_length
        self.label_length = label_length

        self.queue_size = queue_size

        self.train_queue = multiprocessing.Queue(self.queue_size)
        self.test_queue = multiprocessing.Queue(self.queue_size)

        # specify random seed so the result is repeatable
        self.random_seed = random_seed
        np.random.seed(self.random_seed)

    # helper
    def train_process(self, scrambleFlag):
        collection_name = self.collection_prefix + "_train"
        client = MongoClient(self.mongo_url)
        database = client[self.db_name]
        collection = database[collection_name]

        train_record_count = collection.count()
        last_record_index = train_record_count - (self.data_length + self.label_length)

        if scrambleFlag:
            while True:
                index = np.random.randint(0, last_record_index)

                data = collection.find({"index": {"$gt": index - 1}}).sort("index", pymongo.ASCENDING).limit(self.data_length)
                label = collection.find({"index": {"$gt": index + self.data_length - 1}}).sort("index", pymongo.ASCENDING).limit(self.label_length)
                
                data, label = WindowNormalizer.percentage_normalization(data, label)
                
                self.train_queue.put(tuple((data, label)))
        else:
            index = 0
            while True:
                data = collection.find({"index": {"$gt": index - 1}}).sort("index", pymongo.ASCENDING).limit(self.data_length)
                label = collection.find({"index": {"$gt": index + self.data_length - 1}}).sort("index", pymongo.ASCENDING).limit(self.label_length)

                data, label = WindowNormalizer.percentage_normalization(data, label)

                self.train_queue.put(tuple((data, label)))
                index += 1
                if index == last_record_index + 1:
                    index = 0

    def test_process(self, scrambleFlag):
        collection_name = self.collection_prefix + "_test"
        client = MongoClient(self.mongo_url)
        database = client[self.db_name]
        collection = database[collection_name]

        train_record_count = collection.count()
        last_record_index = train_record_count - (self.data_length + self.label_length)

        if scrambleFlag:
            while True:
                index = np.random.randint(0, last_record_index)

                data = collection.find({"index": {"$gt": index - 1}}).sort("index", pymongo.ASCENDING).limit(self.data_length)
                label = collection.find({"index": {"$gt": index + self.data_length - 1}}).sort("index", pymongo.ASCENDING).limit(self.label_length)
                
                data, label = WindowNormalizer.percentage_normalization(data, label)
                
                self.train_queue.put(tuple((data, label)))
        else:
            index = 0
            while True:
                data = collection.find({"index": {"$gt": index - 1}}).sort("index", pymongo.ASCENDING).limit(self.data_length)
                label = collection.find({"index": {"$gt": index + self.data_length - 1}}).sort("index", pymongo.ASCENDING).limit(self.label_length)

                data, label = WindowNormalizer.percentage_normalization(data, label)

                self.train_queue.put(tuple((data, label)))
                index += 1
                if index == last_record_index + 1:
                    index = 0

    # interface
    def start_load_train(self, scrambleFlag=True):
        train_loader = multiprocessing.Process(target=self.train_process)
        train_loader.daemon = True
        train_loader.start()

    def start_load_test(self, scrambleFlag=True):
        test_loader = multiprocessing.Process(target=self.test_process)
        test_loader.daemon = True
        test_loader.start()

    def load_train_sample(self):
        data, label = self.train_queue.get()
        return (data, label)

    def load_test_sample(self):
        data, label = self.test_queue.get()
        return (data, label)
    