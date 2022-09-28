import os
train_path = './csvTestData.csv'

window_size=90

import numpy as np
import csv

recordings=[] # 存放数据

with open(train_path,'r',newline='') as stream:
    reader = csv.reader(stream)
    for row in reader:
        for i in range(1, len(row)):
            row[i] = float(row[i])
        recordings.append(row)

num_channels = int((len(recordings[0]) - 1) / window_size)
possible_labels = list(sorted(set(item[0] for item in recordings)))
print('num_channels = %d' % num_channels)
print("possible_labels = %s" % possible_labels)

labels = np.full([len(recordings), len(possible_labels)], False, dtype=bool)
samples = np.zeros([len(recordings), num_channels, window_size])
print('labels ' , labels.shape)
print("samples " , samples.shape)

for sample_id, sample in enumerate(recordings):
            labels[sample_id][possible_labels.index(sample[0])] = True
            for channel in range(num_channels):
                start = 1 + channel * window_size
                end = 1 + (channel + 1) * window_size
                samples[sample_id][channel] = sample[start:end]

print("samples\n " , samples)
