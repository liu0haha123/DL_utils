import paddle
import paddle.fluid as fluid

from paddle.dataset import imdb
import numpy as np
import sys
import math

CLASS_DIM = 2     #情感分类的类别数
EMB_DIM = 128     #词向量的维度
HID_DIM = 512     #隐藏层的维度
STACKED_NUM = 3   #LSTM双向栈的层数
BATCH_SIZE = 128  #batch的大小
word_dict = paddle.dataset.imdb.word_dict()

def textConv(data,input_dim,classes_nums,emb_dim,hidden_dim):
    emb = fluid.embedding(input=data,size=[input_dim,emb_dim],is_sparse=True)
    Conv3 = fluid.nets.layers.sequence_conv(
        input=emb,
        num_filters=hidden_dim,
        filter_size=3,
        act="tanh",
        pool_type = "sqrt"
    )
    Conv4 = fluid.nets.layers.sequence_conv(
        input=emb,
        num_filters=hidden_dim,
        filter_size=4,
        act="tanh",
        pool_type = "sqrt"
    )
    Conv5 = fluid.nets.layers.sequence_conv(
        input=emb,
        num_filters=hidden_dim,
        filter_size=3,
        act="tanh",
        pool_type = "sqrt"
    )
    predict = fluid.layers.fc(
        input=[Conv3,Conv4,Conv5],
        act="softmax",
        size=classes_nums
    )

    return predict

def BiLSTM_new(data,input_dim,hidden_dim,emb_dim,classes_nums,stacked_num):
    emb = fluid.embedding(input=data,size=[input_dim,emb_dim],is_sparse=True)
    fc1 = fluid.layers.fc(input=emb,size=hidden_dim)
    #LSTM层
    lstm1,cell1 = fluid.layers.dynamic_lstm(input=fc1,size=hidden_dim)

    inputs = [fc1,lstm1]
    #其余的所有栈结构
    for i in range(2, stacked_num + 1):
        fc = fluid.layers.fc(input=inputs, size=hidden_dim)
        lstm, cell = fluid.layers.dynamic_lstm(
            input=fc, size=hidden_dim, is_reverse=(i % 2) == 0)
        inputs = [fc, lstm]

    #池化层
    fc_last = fluid.layers.sequence_pool(input=inputs[0], pool_type='max')
    lstm_last = fluid.layers.sequence_pool(input=inputs[1], pool_type='max')

    #全连接层，softmax预测
    prediction = fluid.layers.fc(
        input=[fc_last, lstm_last], size=classes_nums, act='softmax')
    return prediction

def inference_program(word_dict):
    data = fluid.data(
        name="words", shape=[None], dtype="int64", lod_level=1)
    dict_dim = len(word_dict)
    net = textConv(data, dict_dim, CLASS_DIM, EMB_DIM, HID_DIM)
    # net = stacked_lstm_net(data, dict_dim, CLASS_DIM, EMB_DIM, HID_DIM, STACKED_NUM)
    return net

def train_program(prediction):
    label = fluid.data(name="label",shape=[None,1],dtype="int64")
    loss = fluid.layers.cross_entropy(prediction,label)
    avg_loss = fluid.layers.mean(loss)
    accuracy = fluid.layers.accuracy(input=prediction,label=label)
    return [avg_loss,accuracy]

def optimizer_func():
    return fluid.optimizer.Adagrad(learning_rate=0.002)
use_cuda = True  #在cpu上进行训练
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

print("Loading IMDB word dict....")
word_dict = paddle.dataset.imdb.word_dict()

print ("Reading training data....")
train_reader = fluid.io.batch(
    fluid.io.shuffle(
        paddle.dataset.imdb.train(word_dict), buf_size=25000),
    batch_size=BATCH_SIZE)
print("Reading testing data....")
test_reader = fluid.io.batch(
    paddle.dataset.imdb.test(word_dict), batch_size=BATCH_SIZE)
# 过程式写法的训练过程
exe = fluid.Executor(place)
prediction = inference_program(word_dict)
[avg_cost, accuracy] = train_program(prediction)#训练程序
sgd_optimizer = optimizer_func()#训练优化函数
sgd_optimizer.minimize(avg_cost)