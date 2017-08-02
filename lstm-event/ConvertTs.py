# -*- coding:utf-8 -*-

import sys
sys.path.append("../tool")
from MongoRecordConvertor import MongoRecordConvertor

def handler(record):
    
    return result

if __name__ == "__main__":
    convertor = MongoRecordConvertor(handler, "tploader", "tploader")
    convertor.start()
