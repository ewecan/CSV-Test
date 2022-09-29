
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
trainlabels = np.full([len(trainData), len(possible_labels)], False, dtype=bool)
trainsamples = np.zeros([len(trainData), num_channels, window_size])
print('labels ' , trainlabels.shape)
print("trainsamples " , trainsamples.shape)

for sample_id, sample in enumerate(trainData):
    trainlabels[sample_id][possible_labels.index(sample[0])] = True
    for channel in range(num_channels):
        start = 1 + channel * window_size
        end = 1 + (channel + 1) * window_size
        trainsamples[sample_id][channel] = sample[start:end]
# trainsamples=np.reshape(trainsamples, (len(recordings)*num_channels* window_size))
trainsamples=np.reshape(trainsamples,(-1,window_size,num_channels,1))
print("trainsamples " , trainsamples.shape)


print("############################解析测试数据####################################")
testlabels = np.full([len(testData), len(possible_labels)], False, dtype=bool)
testsamples = np.zeros([len(testData), num_channels, window_size])
print('labels ' , testlabels.shape)
print("testsamples " , testsamples.shape)

for sample_id, sample in enumerate(testData):
    testlabels[sample_id][possible_labels.index(sample[0])] = True
    for channel in range(num_channels):
        start = 1 + channel * window_size
        end = 1 + (channel + 1) * window_size
        testsamples[sample_id][channel] = sample[start:end]
# samples=np.reshape(samples, (len(recordings)*num_channels* window_size))
testsamples=np.reshape(testsamples,(-1,window_size,num_channels,1))
print("testsamples " , testsamples.shape)

print("##############################洗牌##################################")
from sklearn.utils import shuffle
trainsamples, trainlabels = shuffle(trainsamples, trainlabels, random_state=41)

print(trainlabels)

from tensorflow.keras import utils
trainlabels = utils.to_categorical(trainlabels)

testactual  = testlabels
testlabels  = utils.to_categorical(testlabels) 

print(testactual)

print("############################构建网络模型################################")
import tensorflow as tf
from tensorflow.keras import models, layers, losses
input_shape = (window_size,num_channels,1) 
num_classes = len(possible_labels)

print("输入标签数量:",num_classes)

## Build Model
inputs = layers.Input(shape=input_shape)
# 1st Convolutional layer
x = layers.Conv2D(16, 3, activation = 'relu', padding = 'same')(inputs)
# Fully Connected layer        
x = layers.Flatten()(x)
x = layers.Dense(32)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs=inputs, outputs=outputs)

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(trainsamples, trainlabels, batch_size=10, epochs=30, validation_data=(testsamples, testlabels))

print("############################模型评估########################")
score = model.evaluate(testsamples, testlabels)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])