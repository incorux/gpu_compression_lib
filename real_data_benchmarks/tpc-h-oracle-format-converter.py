import csv
import sys
import io
import struct
import datetime
import time


if __name__ == "__main__":

    fin = "lineitem.tbl"
    f_out = { 0: 'ORDERKEY', 3: 'LINENUMBER', 11: 'COMMITDATE', 5: 'EXTENDEDPRICE'}

    f_converter = { 
            0: lambda x: int(x), 
            3: lambda x: int(x), 
            11: lambda x: int(time.mktime(datetime.datetime.strptime(x, "%Y-%m-%d").timetuple())),
            5: lambda x: int(float(x) * 100) 
    }
    fds = {}
    for k in f_out: fds[k] = open(f_out[k] + ".bin", "wb")

    io.BufferedIOBase()

    with open(fin) as f:
        data = csv.reader(f, delimiter="|")
        for record in data:
            for col in f_out:
                fds[col].write(struct.pack('=i', f_converter[col](record[col])))

    for k in fds:
        fds[k].close()
