import numpy as np
import csv
import pandas as pd
from scipy import signal
from scipy.signal import butter, lfilter, freqz
from matplotlib import pyplot as plt
import math as m
import argparse

''' 
This program read in a raw data file (.txt) and filter it using a bandpass filter,
then write the filtered data to a csv file. (adapted from pretest.ipynb from NTUEE)
'''

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("input_file", help="input file name")
parser.add_argument("output_file", help="output file name")
args = parser.parse_args()

# Check arguments
if args.input_file == None:
    print("Error: input file name is not specified.")
    exit()
if args.output_file == None:
    print("Error: output file name is not specified.")
    exit()

in_path = args.input_file
out_path = args.output_file
    
# Filters
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs  # nyquist frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def notch_filter(data, f0, Q, fs):
    b, a = signal.iirnotch(f0, Q, fs)
    y = lfilter(b, a, data)
    return y

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Full wave rectification
# calculate absolute value
def FilterData(data, sampling_rate):
    #not1 = notch_filter(data, 60, 30, sampling_rate)
    #high1 = butter_highpass_filter(data, 10, sampling_rate, order=6)
    high1 = butter_bandpass_filter(data,20,400,sampling_rate,order = 5)
    #rec1 = abs(high1)
    #lpass1 = butter_lowpass_filter(rec1, 499, sampling_rate, order = 6)
    win = signal.windows.hann(20)
    filtered = signal.convolve(high1, win, mode='same') / sum(win)
    rec = abs(filtered)
    return rec


count = 0
ch = []

# Read in data from .txt file
with open(in_path) as f:
    for line in f.readlines():
        count += 1
        if count >= 7 and len(line) > 5:
            s = line.split('\t')    # split channel data
            k = s[3].split('\n')
            s[3] = k[0]             # remove '\n'
            s = list(map(int, s))   # convert string to int
            ch.append(list(s))
ch=np.transpose(ch) 
print(np.shape(ch))
f.close


filter_period = 150
fdata = []
segmented_data= []

# Filter data
for j in range(3):
    temp_fdata = []
    for i in range( int(len(ch[0])/filter_period)):
        cur_window = FilterData(ch[j+1][i*filter_period : (i+1)*filter_period], 1000)
        temp_fdata.extend(cur_window)
        
    fdata.append(temp_fdata)

# Write filtered data to csv file
with open(out_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(len(fdata[0])):
        writer.writerow([fdata[0][i], fdata[1][i], fdata[2][i]])

csvfile.close()