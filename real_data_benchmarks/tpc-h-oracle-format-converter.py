import csv
import sys
import io
import struct
import datetime
import time

import concurrent.futures

# 0, L_ORDERKEY identifier Foreign Key to O_ORDERKEY
# 1, L_PARTKEY identifier Foreign key to P_PARTKEY, first part of the compound Foreign Key to (PS_PARTKEY, PS_SUPPKEY) with L_SUPPKEY
# 2, L_SUPPKEY Identifier Foreign key to S_SUPPKEY, second part of the compound Foreign Key to (PS_PARTKEY, TPC Benchmark TM H Standard Specification Revision 2.17.1 Page 16PS_SUPPKEY) with L_PARTKEY
# 3, L_LINENUMBER integer
# 4, L_QUANTITY decimal
# 5, L_EXTENDEDPRICE decimal
# 6, L_DISCOUNT decimal
# 7, L_TAX decimal
# 8, L_RETURNFLAG fixed text, size 1
# 9, L_LINESTATUS fixed text, size 1
# 10, L_SHIPDATE date
# 11, L_COMMITDATE date
# 12, L_RECEIPTDATE date
# 13, L_SHIPINSTRUCT fixed text, size 25
# 14, L_SHIPMODE fixed text, size 10
# 15, L_COMMENT variable text size 44

fin = "lineitem.tbl"
f_out = { 4: 'QUANTITY', 1: 'PARTKEY', 10: 'SHIPDATE', 6: 'DISCOUNT', 0: 'ORDERKEY', 3: 'LINENUMBER', 11: 'COMMITDATE', 5: 'EXTENDEDPRICE'}
columns_ids = f_out.keys()


def extrac_to_binary(output_id):
    f_converter = { 
            0: lambda x: int(x), 
            1: lambda x: int(x), 
            3: lambda x: int(x), 
            10: lambda x: int(time.mktime(datetime.datetime.strptime(x, "%Y-%m-%d").timetuple())),
            11: lambda x: int(time.mktime(datetime.datetime.strptime(x, "%Y-%m-%d").timetuple())),
            5: lambda x: int(float(x) * 100),
            6: lambda x: int(float(x) * 100),
            4: lambda x: int(float(x) * 100),
            }
    fd_out = open(f_out[output_id] + ".bin", "wb")
    io.BufferedIOBase()
    func = f_converter[output_id]
    with open(fin) as f:
        data = csv.reader(f, delimiter="|")
        for record in data:
            fd_out.write(struct.pack('=i', func(record[output_id])))

    fd_out.close()
    return output_id

if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor(len(columns_ids)) as executor:
        for done in executor.map(extrac_to_binary, columns_ids):
            print 'Done file %s' % (f_out[done],)
