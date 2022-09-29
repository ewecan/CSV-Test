import os
train_path = './csvTrainData.csv'
test_path = './csvTestData.csv'

window_size=90

import numpy as np
import csv

print("############################导入数据####################################")
trainData=[]    # 训练数据
testData=[]     # 测试数据
with open(train_path,'r',newline='') as stream:
    reader = csv.reader(stream)
    for row in reader:
        for i in range(1, len(row)):
            row[i] = float(row[i])
        trainData.append(row)
print("训练数据 长度:", len(trainData))

with open(test_path,'r',newline='') as stream:
    reader = csv.reader(stream)
    for row in reader:
        for i in range(1, len(row)):
            row[i] = float(row[i])
        testData.append(row)
print("测试数据 长度:",  len(testData))


print("##############################解析数据####################################")
num_channels = int((len(trainData[0]) - 1) / window_size)
possible_labels = list(sorted(set(item[0] for item in trainData)))
print('采集通道数: %d' % num_channels)
print("标签: %s" % possible_labels)

print("############################解析训练数据####################################")
labels = np.full([len(trainData), len(possible_labels)], False, dtype=bool)
trainsamples = np.zeros([len(trainData), num_channels, window_size])
print('labels ' , labels.shape)
print("trainsamples " , trainsamples.shape)

for sample_id, sample in enumerate(trainData):
    labels[sample_id][possible_labels.index(sample[0])] = True
    for channel in range(num_channels):
        start = 1 + channel * window_size
        end = 1 + (channel + 1) * window_size
        trainsamples[sample_id][channel] = sample[start:end]
# trainsamples=np.reshape(trainsamples, (len(recordings)*num_channels* window_size))
trainsamples=np.reshape(trainsamples,(-1,window_size,num_channels,1))
print("trainsamples " , trainsamples.shape)


print("############################解析测试数据####################################")
labels = np.full([len(testData), len(possible_labels)], False, dtype=bool)
testsamples = np.zeros([len(testData), num_channels, window_size])
print('labels ' , labels.shape)
print("testsamples " , testsamples.shape)

for sample_id, sample in enumerate(testData):
    labels[sample_id][possible_labels.index(sample[0])] = True
    for channel in range(num_channels):
        start = 1 + channel * window_size
        end = 1 + (channel + 1) * window_size
        testsamples[sample_id][channel] = sample[start:end]
# samples=np.reshape(samples, (len(recordings)*num_channels* window_size))
testsamples=np.reshape(testsamples,(-1,window_size,num_channels,1))
print("testsamples " , testsamples.shape)

