# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import re


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--filename", type=str, help="The name of log which need to analysis.")
    parser.add_argument(
        "--keyword", type=str, help="Keyword to specify analysis data")
    parser.add_argument(
        "--separator", type=str, default=" ", help="Separator of different field in log")
    parser.add_argument(
        '--position', type=int, default=-1, help='The position of data field')
    parser.add_argument(
        '--batch_size', type=int, help='batch size')
    parser.add_argument(
        '--skip_steps', type=int, default=0, help='The number of steps to be skipped')
    parser.add_argument(
        '--mode', type=int, default=0, help='Analysis mode, 0')
    args = parser.parse_args()
    return args


class TimeAnalyzer(object):
    def __init__(self, filename, keyword=None, separator=" ", position=-1):
        if filename is None:
            raise Exception("Please specify the filename!")

        if keyword is None:
            raise Exception("Please specify the keyword!")

        self.filename = filename
        self.keyword = keyword
        self.separator = separator
        self.position = position
        self.records = None
        self._distil()

    def _distil(self):
        self.records = []

        f = open(self.filename, "r")
        lines = f.readlines()
        for line in lines:
            if line.find(self.keyword) == -1:
                continue

            line = line.strip().replace("\t", self.separator)
            line = re.sub(r"\s+", self.separator, line)
            items = line.split(self.separator)
            clean_items = []
            for item in items:
                if item != self.separator:
                    clean_items.append(item)

            if self.position >= 0 and self.position < len(clean_items):
                position = self.position
            else:
                position = 0
                for item in clean_items:
                    position += 1
                    if item == self.keyword:
                        break
            #print(position)
            #print(clean_items)
            #print(clean_items[position].replace("s", ""))
            self.records.append(float(clean_items[position].replace("s", "")))
            
        if len(self.records) <= 0:
            raise Exception("No items in %s!" % (self.filename))

    def analysis(self, batch_size, skip_steps=0, mode=0):
        if batch_size <= 0:
            raise ValueError("batch_size should larger than 0.")

        sum_of_records = 0
        sum_of_records_skipped = 0
        skip_min = self.records[skip_steps]
        skip_max = self.records[skip_steps]

        count = len(self.records)
        for i in range(count):
            sum_of_records += self.records[i]
            if i >= skip_steps:
                sum_of_records_skipped += self.records[i]
                if self.records[i] < skip_min:
                    skip_min = self.records[i]
                if self.records[i] > skip_max:
                    skip_max = self.records[i]

        avg_of_records = sum_of_records / float(count)
        avg_of_records_skipped = sum_of_records_skipped / float(count - skip_steps)

        if mode == 1:
            avg_of_records = float(1) / avg_of_records
            avg_of_records_skipped = float(1) / avg_of_records_skipped
            skip_min = float(1) / skip_min
            skip_max = float(1) / skip_max
            
        print("average latency of %d steps, skip 0 step:" % (count))
        print("\tAvg: %.3f s/step" % (avg_of_records))
        print("\tFPS: %.3f samples/s" % (batch_size / avg_of_records))
        if skip_steps > 0:
            print("average latency of %d steps, skip %d steps:" % (count, skip_steps))
            print("\tAvg: %.3f s/step" % (avg_of_records_skipped))
            print("\tMin: %.3f s/step" % (skip_min))
            print("\tMax: %.3f s/step" % (skip_max))
            print("\tFPS: %.3f samples/s" % (batch_size / avg_of_records_skipped))


if __name__ == "__main__":
    args = parse_args()
    if args.filename is None:
        # Test the script file
        analyzer = TimeAnalyzer("log", "time:", " ", -1)
        analyzer.analysis(32, 10, 0)
    else:
        analyzer = TimeAnalyzer(args.filename, args.keyword, args.separator, args.position)
        analyzer.analysis(args.batch_size, args.skip_steps, args.mode)
