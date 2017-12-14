# -*- coding:utf-8 -*-

import sys
sys.path.append("../data-preprocessing")
import WindowNormalizer

import time
import multiprocessing
import numpy as np
import pymongo
from pymongo import MongoClient

class SampleLoader(object):
    def __init__(self, db_name, collection_prefix, test_index_offset=0, mongo_url="mongodb://localhost:27017", batch_size=4, data_length=5, label_length=1, random_seed=int(9527), queue_size=32):
        self.db_name = db_name
        self.collection_prefix = collection_prefix
        self.mongo_url = mongo_url
        self.test_index_offset = test_index_offset

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

                data = collection.find({"i": {"$gt": index - 1}}).sort("i", pymongo.ASCENDING).limit(self.data_length)
                label = collection.find({"i": {"$gt": index + self.data_length - 1}}).sort("i", pymongo.ASCENDING).limit(self.label_length)

                data_array, label_array = self.convert_array(data, label)

                self.train_queue.put(tuple((data_array, label_array)))
        else:
            index = 0
            while True:
                data = collection.find({"i": {"$gt": index - 1}}).sort("i", pymongo.ASCENDING).limit(self.data_length)
                label = collection.find({"i": {"$gt": index + self.data_length - 1}}).sort("i", pymongo.ASCENDING).limit(self.label_length)

                data_array, label_array = self.convert_array(data, label)
                
                self.train_queue.put(tuple((data_array, label_array)))
                index += 1
                if index == last_record_index + 1:
                    index = 0

    def test_process(self, scrambleFlag):
        collection_name = self.collection_prefix + "_test"
        client = MongoClient(self.mongo_url)
        database = client[self.db_name]
        collection = database[collection_name]

        train_record_count = collection.count()
        last_record_index = train_record_count - (self.data_length + self.label_length) + self.test_index_offset

        if scrambleFlag:
            while True:
                index = np.random.randint(self.test_index_offset, last_record_index)
                # print("test index: {0}".format(index))

                data = collection.find({"i": {"$gt": index - 1}}).sort("i", pymongo.ASCENDING).limit(self.data_length)
                label = collection.find({"i": {"$gt": index + self.data_length - 1}}).sort("i", pymongo.ASCENDING).limit(self.label_length)
                
                data_array, label_array = self.convert_array(data, label)

                self.test_queue.put(tuple((data_array, label_array)))
        else:
            index = 0
            while True:
                data = collection.find({"i": {"$gt": index - 1}}).sort("i", pymongo.ASCENDING).limit(self.data_length)
                label = collection.find({"i": {"$gt": index + self.data_length - 1}}).sort("i", pymongo.ASCENDING).limit(self.label_length)

                data_array, label_array = self.convert_array(data, label)
                
                self.test_queue.put(tuple((data_array, label_array)))
                index += 1
                if index == last_record_index + 1:
                    index = 0

    def convert_array(self, data, label):
        data_array = []
        label_array = []

        day_minutes = []
        week_days = []
        
        err_rates = []
        trans_counts = []

        tm_smoothness = []
        avg_tms = []

        for rec in data:
            ts_tuple = time.localtime(rec["ts"])

            # normalization on the fly
            day_minutes.append(float(ts_tuple.tm_hour) * 60. + float(ts_tuple.tm_min) / (24. * 60.))
            week_days.append(float(ts_tuple.tm_wday) / 7.)

            # no need to normalize
            err_rates.append(float(rec["err"]) / float(rec["cnt"]))
            tm_smoothness.append(float(rec["avgTm"]) / float(rec["maxTm"]))

            # need offline normalization
            trans_counts.append(float(rec["cnt"]))
            avg_tms.append(float(rec["avgTm"]))

        for rec in label:
            ts_tuple = time.localtime(rec["ts"])

            # normalization on the fly
            day_minutes.append(float(ts_tuple.tm_hour) * 60. + float(ts_tuple.tm_min) / (24. * 60.))
            week_days.append(float(ts_tuple.tm_wday) / 7.)

            # no need to normalize
            err_rates.append(float(rec["err"]) / float(rec["cnt"]))
            tm_smoothness.append(float(rec["avgTm"]) / float(rec["maxTm"]))

            # need offline normalization
            trans_counts.append(float(rec["cnt"]))
            avg_tms.append(float(rec["avgTm"]))

        if len(trans_counts) == 0:
            return ([], [])

        trans_counts = WindowNormalizer.percentage_normalization(trans_counts)
        avg_tms = WindowNormalizer.percentage_normalization(avg_tms)
        
        for i in range(self.data_length):
            sample = np.array([day_minutes[i], week_days[i], err_rates[i], trans_counts[i], tm_smoothness[i], avg_tms[i]])
            data_array.append(sample)

        for i in range(self.data_length, self.data_length + self.label_length):
            result = np.array([err_rates[i], trans_counts[i], tm_smoothness[i], avg_tms[i]])
            label_array.append(result)

        return (data_array, label_array)

    # interface
    def start_load_train(self, scrambleFlag=True):
        train_loader = multiprocessing.Process(target=self.train_process, args=(scrambleFlag, ))
        train_loader.daemon = True
        train_loader.start()

    def start_load_test(self, scrambleFlag=True):
        test_loader = multiprocessing.Process(target=self.test_process, args=(scrambleFlag, ))
        test_loader.daemon = True
        test_loader.start()

    def load_train_sample(self):
        datas, labels = [], []
        for i in range(self.batch_size):
            data, label = self.train_queue.get()
            datas.append(data)
            labels.append(label)
        return (datas, labels)

    def load_test_sample(self):
        datas, labels = [], []
        for i in range(self.batch_size):
            data, label = self.test_queue.get()
            datas.append(data)
            labels.append(label)
        return (datas, labels)
    