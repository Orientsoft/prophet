import multiprocessing
from pymongo import MongoClient

class MongoRecordConvertor(object):
    def __init__(self, db_name, collection_name, mongo_url='mongodb://localhost:27017', queue_size=64, bulk_size=1024, in_place=False):
        self.db_name = db_name
        self.collection_name = collection_name
        self.mongo_url = mongo_url
        self.bulk_size = bulk_size
        self.in_place = in_place

        client = MongoClient(self.mongo_url)
        db = client[self.db_name]
        collection = db[self.collection_name]
        self.record_count = collection.count()

        self.queue_size = queue_size
        self.read_queue = multiprocessing.Queue(self.queue_size)
        self.write_queue = multiprocessing.Queue(self.queue_size)

    def read_process(self):
        client = MongoClient(self.mongo_url)
        db = client[self.db_name]
        collection = db[self.collection_name]

        for record in collection.find():
            self.read_queue.put(record)

    def convert_process(self):
        for i in self.record_count:
            record = self.read_queue.get()
            # TODO : convert ts
            self.write_queue.put(record)

    def write_process(self):
        client = MongoClient(self.mongo_url)
        db = client[self.db_name]
        collection = db[self.collection_name]

        bulk = []
        record_count = 0

        for i in self.record_count:
            record = self.read_queue.get()
            bulk.append(record)
            record_count += 1

            if record_count == self.bulk_size:
                collection.insert_many(bulk)

                bulk.clear()
                record_count = 0

    def start(self):