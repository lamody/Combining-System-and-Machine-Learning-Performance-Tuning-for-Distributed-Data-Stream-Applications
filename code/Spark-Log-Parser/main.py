import os
import sys

from spark_run import SparkRun

"""
def parse_dir(path):
    for dirpath, subdirs, files in os.walk(path):
        for subdirname in subdirs:
            parse_dir(os.path.join(dirpath, subdirname))

        for filename in files:
            try:
                a = SparkRun(os.path.join(dirpath, filename))
            except ValueError:
                continue
            a.correlate()
            # Link block managers and executors
            report = a.generate_report()
            print("The parser is generating report...")
            open(os.path.join(dirpath, a.get_app_name() + ".txt"), "w").write(report)
"""

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print "Usage python <main.py> <log_file> <out file>"
        exit(1)
    else:
        PATH = os.path.join(sys.argv[1])
        out = os.path.join(sys.argv[2])

    a = SparkRun(PATH)
    a.correlate()
    # Link block managers and executors
    report = a.generate_report()
    open(out, "w").write(report)
    # open(os.path.join(dirpath, a.get_app_name() + ".txt"), "w").write(report)
