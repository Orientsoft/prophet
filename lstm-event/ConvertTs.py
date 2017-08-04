# -*- coding:utf-8 -*-

import sys
sys.path.append("../tool")
from MongoRecordConvertor import MongoRecordConvertor

from datetime import datetime

# { "_id" : ObjectId("59060a041789bdc9367f54df"), "check_time" : "2017-05-01 00:00:02", "host_port" : "98.10.2.216:80", "check_type" : "tploader", "check_state" : "ACTIVE", "AcctNo1" : "********", "AcctNo2" : "********", "Amt1" : "50.0", "Amt2" : "50.0", "Brc" : "10187", "Ccy1" : "01", "Ccy2" : "01", "ChannelId" : "73", "CheckFlag" : "0", "Flag1" : "6", "Flag2" : "6", "Flag6" : "3", "FrntNo" : "FRNT0001", "IoFlag" : "0", "PkgType" : "RspPkg", "Teller" : "TYZF", "TermDate" : "20170501", "TermSeq" : "18553954", "TermTime" : "000913", "TranCode" : "801001", "TranKind" : "005", "PkgStartTime" : "05-01@00:09:13:599270", "TransCode" : "tp801001", "AcctName1" : "***", "AcctNo4" : "********", "AuthCode" : "0", "Bal1" : "53.030000", "Bal2" : "53.030000", "Brc1" : "10161", "Brc2" : "10187", "BrcAttr0" : "1", "BrcLvl0" : "5", "BrcName" : "**路支行", "BrcName0" : "*部**室", "BrcType0" : "08", "FileFlag" : "0", "FrontCtrlFlag" : "6", "PT1" : "贷@********@@@@@@@@@@@@@@@@50.00", "PT2" : "借@********@@@@@@@@@@@@@@@@50.00", "PubBrcPhone" : "4600166", "PubCcpcNo" : "8210", "RspCode" : "000000", "RspTime" : "05-01@00:09:14:086439", "SerSeqNo" : "70636", "SubAcct4" : "********", "TellerLvl0" : "0", "TranDate" : "2017-05-01", "TranName" : "****系统**交易", "TranTime" : "000913", "UseTime" : 487 }

def handler(record):
    try:
        record["check_time"] = datetime.strptime(record["check_time"], "%Y-%m-%d %H:%M:%S")
        record["PkgStartTime"] = datetime.strptime(record["PkgStartTime"], "%m-%d@%H:%M:%S:%f")
        record["RspTime"] = datetime.strptime(record["RspTime"], "%m-%d@%H:%M:%S:%f")
    except Exception as ex:
        err = ex
    
    try:
        record["PkgStartTime"] = record["PkgStartTime"].replace(2016)
        record["RspTime"] = record["RspTime"].replace(2016)
    except Exception as ex:
        print(record)
    
    return record

if __name__ == "__main__":
    convertor = MongoRecordConvertor(handler, "tploader", "tploader")
    convertor.start()
