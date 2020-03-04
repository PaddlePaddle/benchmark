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
import json
import re
import traceback


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--filename", type=str, help="The name of log which need to analysis.")
    parser.add_argument(
        "--log_with_profiler", type=str, help="The path of train log with profiler")
    parser.add_argument(
        "--profiler_path", type=str, help="The path of profiler timeline log.")
    parser.add_argument(
        "--keyword", type=str, help="Keyword to specify analysis data")
    parser.add_argument(
        "--separator", type=str, default=" ", help="Separator of different field in log")
    parser.add_argument(
        '--position', type=int, default=-1, help='The position of data field')
    parser.add_argument(
        '--range', type=str, default="", help='The range of data field to intercept')
    parser.add_argument(
        '--base_batch_size', type=int, help='base_batch size on gpu')
    parser.add_argument(
        '--skip_steps', type=int, default=0, help='The number of steps to be skipped')
    parser.add_argument(
        '--model_mode', type=int, default=0, help='Analysis mode, 0')
    parser.add_argument(
        '--model_name', type=str, default=0, help='training model_name, transformer_base')
    parser.add_argument(
        '--run_mode', type=str, default="sp", help='multi process or single process')
    parser.add_argument(
        '--index', type=int, default=1, help='{1: speed, 2:mem, 3:profiler, 6:max_batch_size}')
    parser.add_argument(
        '--gpu_num', type=int, default=1, help='nums of training gpus')
    args = parser.parse_args()
    return args


def _is_number(num):
    pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
    result = pattern.match(num)
    if result:
        return True
    else:
        return False


class TimeAnalyzer(object):
    def __init__(self, filename, keyword=None, separator=" ", position=-1, range="-1"):
        if filename is None:
            raise Exception("Please specify the filename!")

        if keyword is None:
            raise Exception("Please specify the keyword!")

        self.filename = filename
        self.keyword = keyword
        self.separator = separator
        self.position = position
        self.range = range
        self.records = None
        self._distil()

    def _distil(self):
        self.records = []
        with open(self.filename, "r") as f_object:
            lines = f_object.readlines()
            for line in lines:
                if self.keyword not in line:
                    continue
                try:
                    line = line.strip()
                    if self.separator:
                        result = line.split(self.separator)[self.position]
                    else:
                        result = line.split()[self.position]
                    if not self.range:
                        result = result[0:]
                    elif _is_number(self.range):
                        result = result[0: int(self.range)]
                    else:
                        result = result[int(self.range.split(":")[0]): int(self.range.split(":")[1])]
                    self.records.append(float(result))
                except Exception as exc:
                    print("line is: {}; separator={}; position={}".format(line, self.separator, self.position))

        print("Extract {} records: separator={}; position={}".format(len(self.records), self.separator, self.position))

    def _get_fps(self, mode, batch_size, gpu_num, avg_of_records):
        if mode == 0:
            # s/step -> samples/s
            fps = (batch_size * gpu_num) / avg_of_records
            unit = "samples/s"
        elif mode == 1:
            # steps/s -> steps/s
            fps = avg_of_records
            unit = "steps/s"
        elif mode == 2:
            # s/step -> steps/s
            fps = 1 / avg_of_records
            unit = "steps/s"
        elif mode == 3:
            # steps/s -> samples/s
            fps = batch_size * gpu_num * avg_of_records
            unit = "samples/s"
        else:
            ValueError("Unsupported analysis mode.")

        return fps, unit

    def analysis(self, batch_size, gpu_num=1, skip_steps=0, mode=0):
        if batch_size <= 0:
            print("base_batch_size should larger than 0.")
            return 0

        if len(self.records) <= 0:
            print("no records")
            return 0

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

        fps, fps_unit = self._get_fps(mode, batch_size, gpu_num, avg_of_records)
        fps_skipped, _ = self._get_fps(mode, batch_size, gpu_num, avg_of_records_skipped)
        if mode == 1 or mode == 3:
            print("average latency of %d steps, skip 0 step:" % count)
            print("\tAvg: %.3f steps/s" % avg_of_records)
            print("\tFPS: %.3f %s" % (fps, fps_unit))
            if skip_steps > 0:
                print("average latency of %d steps, skip %d steps:" % (count, skip_steps))
                print("\tAvg: %.3f steps/s" % avg_of_records_skipped)
                print("\tMin: %.3f steps/s" % skip_min)
                print("\tMax: %.3f steps/s" % skip_max)
                print("\tFPS: %.3f %s" % (fps_skipped, fps_unit))
        elif mode == 0 or mode == 2:
            print("average latency of %d steps, skip 0 step:" % count)
            print("\tAvg: %.3f s/step" % avg_of_records)
            print("\tFPS: %.3f %s" % (fps, fps_unit))
            if skip_steps > 0:
                print("average latency of %d steps, skip %d steps:" % (count, skip_steps))
                print("\tAvg: %.3f s/step" % avg_of_records_skipped)
                print("\tMin: %.3f s/step" % skip_min)
                print("\tMax: %.3f s/step" % skip_max)
                print("\tFPS: %.3f %s" % (fps_skipped, fps_unit))

        return round(fps_skipped, 3)


if __name__ == "__main__":
    args = parse_args()
    run_info = dict()
    run_info["log_file"] = args.filename
    run_info["log_with_profiler"] = args.log_with_profiler
    run_info["profiler_path"] = args.profiler_path
    run_info["model_name"] = args.model_name
    run_info["run_mode"] = args.run_mode
    run_info["index"] = args.index
    run_info["gpu_num"] = args.gpu_num
    run_info["FINAL_RESULT"] = 0

    try:
        if args.index == 1:
            analyzer = TimeAnalyzer(args.filename, args.keyword, args.separator, args.position, args.range)
            run_info["FINAL_RESULT"] = analyzer.analysis(args.base_batch_size, args.gpu_num, args.skip_steps, args.model_mode)
        elif args.index == 3:
            run_info["FINAL_RESULT"] = {}
            records_fo_total = TimeAnalyzer(args.filename, 'Framework overhead', None, 3).records
            records_fo_ratio = TimeAnalyzer(args.filename, 'Framework overhead', None, 5).records
            records_ct_total = TimeAnalyzer(args.filename, 'Computation time', None, 3).records
            records_gm_total = TimeAnalyzer(args.filename, 'GpuMemcpy                Calls', None, 4).records
            records_gm_ratio = TimeAnalyzer(args.filename, 'GpuMemcpy                Calls', None, 6).records
            records_gmas_total = TimeAnalyzer(args.filename, 'GpuMemcpyAsync         Calls', None, 4).records
            records_gms_total = TimeAnalyzer(args.filename, 'GpuMemcpySync          Calls', None, 4).records
            run_info["FINAL_RESULT"]["Framework_Total"] = records_fo_total[0] if records_fo_total else 0
            run_info["FINAL_RESULT"]["Framework_Ratio"] = records_fo_ratio[0] if records_fo_ratio else 0
            run_info["FINAL_RESULT"]["ComputationTime_Total"] = records_ct_total[0] if records_ct_total else 0
            run_info["FINAL_RESULT"]["GpuMemcpy_Total"] = records_gm_total[0] if records_gm_total else 0
            run_info["FINAL_RESULT"]["GpuMemcpy_Ratio"] = records_gm_ratio[0] if records_gm_ratio else 0
            run_info["FINAL_RESULT"]["GpuMemcpyAsync_Total"] = records_gmas_total[0] if records_gmas_total else 0
            run_info["FINAL_RESULT"]["GpuMemcpySync_Total"] = records_gms_total[0] if records_gms_total else 0
        else:
            print("Not support!")
    except Exception:
            traceback.print_exc()
    print("{}".format(json.dumps(run_info)))  # it's required, for the log file path  insert to the database
