# -*- coding:utf-8 -*-

import sys
sys.path.append("../tool")
from MongoStatistics import MongoStatistics

import datetime
import time
import numpy as np

def handler(record_count, read_queue, write_queue):
    index = 0
    currMin = 0

    transCount = 0
    errCount = 0
    maxProcessingTime = 0
    avgProcessingTime = 0.
    ts = time.localtime(0.)

    for i in range(record_count):
        record = read_queue.get()

        if record is None:
            result = {"i": index, "ts": ts, "min": currMin, "cnt": transCount, "maxTm": maxProcessingTime, "avgTm": avgProcessingTime, "err": errCount}
            write_queue.put(result)

            write_queue.put(None)
            return

        try:
            PkgStartTime = record["PkgStartTime"]
            tt = PkgStartTime.timetuple()
            ts = time.mktime(tt)
            minute = np.int(np.rint(ts / 60))

            if (minute != currMin) and (currMin != 0):
                avgProcessingTime /= transCount

                result = {"i": index, "ts": ts, "min": currMin, "cnt": transCount, "maxTm": maxProcessingTime, "avgTm": avgProcessingTime, "err": errCount}
                write_queue.put(result)

                transCount = 1
                maxProcessingTime = record["UseTime"]
                avgProcessingTime = 0.
                if record["RspCode"] != "000000":
                    errCount = 1
                else:
                    errCount = 0

                currMin = minute
                index += 1
            else:
                currMin = minute

                transCount += 1

                if record["RspCode"] != "000000":
                    errCount += 1

                if maxProcessingTime < record["UseTime"]:
                    maxProcessingTime = record["UseTime"]

                avgProcessingTime += record["UseTime"]
        except Exception as ex:
            print("exception: {0}, record: {1}".format(ex, record))

if __name__ == "__main__":
    computer = MongoStatistics(handler, "tploader", "tploader", "tploader-1m")
    computer.start()
