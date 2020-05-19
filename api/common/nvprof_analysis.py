import os
import csv

def AnalysisNvprofResultCSV(file):
    total_time = 0.0
    with open(file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if(row[0] == 'GPU activities'):
                total_time = float(row[2]) / float(row[1]) * 100
                return total_time

    if total_time == 0.0:
        print("Error: No cuda kernel time found by nvprof,Please check.")

