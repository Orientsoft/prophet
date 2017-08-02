# -*- coding:utf-8 -*-

import multiprocessing
from tqdm import tqdm
from pymongo import MongoClient

class MongoRecordConvertor(object):
    def __init__(self, handler, db_name, collection_name, mongo_url='mongodb://localhost:27017', queue_size=64, bulk_size=1000):
        self.handler = handler
        self.db_name = db_name
        self.collection_name = collection_name
        self.mongo_url = mongo_url
        self.bulk_size = bulk_size

        client = MongoClient(self.mongo_url)
        database = client[self.db_name]
        collection = database[self.collection_name]
        self.record_count = collection.count()

        self.queue_size = queue_size
        self.read_queue = multiprocessing.Queue(self.queue_size)
        self.write_queue = multiprocessing.Queue(self.queue_size)

    def read_process(self):
        client = MongoClient(self.mongo_url)
        database = client[self.db_name]
        collection = database[self.collection_name]

        for record in collection.find():
            self.read_queue.put(record)

    def convert_process(self):
        for i in self.record_count:
            record = self.read_queue.get()
            result = self.handler(record)
            self.write_queue.put(result)

    def write_process(self):
        progress_bar = tqdm(total=self.record_count)

        client = MongoClient(self.mongo_url)
        database = client[self.db_name]
        collection = database[self.collection_name]

        record_count = 0
        bulk = collection.initialize_unordered_bulk_op()

        for i in range(self.record_count):
            record = self.read_queue.get()

            bulk.find({'_id': record['_id']}).replaceOne(record)
            record_count += 1

            progress_bar.update(1)

            if (record_count == self.bulk_size) or (i == self.record_count - 1):
                bulk.execute()
                bulk = collection.initialize_unordered_bulk_op()
                record_count = 0

        progress_bar.close()

    def start(self):
        reader = multiprocessing.Process(target=self.read_process)
        reader.daemon = True
        reader.start()

        convertor = multiprocessing.Process(target=self.convert_process)
        convertor.daemon = True
        convertor.start()

        self.write_process()
