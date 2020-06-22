# This script runs all the pipeline steps
import sys
import os

BASE_OUTPUT_PATH = sys.argv[1]
import pipeline_commons


if __name__ == "__main__":
    with open(sys.argv[2], "r") as scenesHandleFile:
        line = scenesHandleFile.readline()
        cnt = 1
        while line:
            print("Line {}: {}".format(cnt, line.strip()))
            line = scenesHandleFile.readline()
            cnt += 1

