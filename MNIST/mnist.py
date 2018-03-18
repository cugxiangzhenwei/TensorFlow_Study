#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#MNIST 数据集相关的常数
INPUT_DATA = 784 #输入层的节点个数，对于MNIST数据集，这个就是等于图片的像素
OUTPUT_NODE = 10 #输出层的节点数。这个等于类别的数目。因为在MNIST数据集中，需要区分的是0~9的10个数字，所以在这里输出层的节点数为10

#配置神经网络的参数。
LAYER1_NODE = 500 #隐藏层节点数。这里使用一个隐藏层的网络结构作为样例,隐藏层有500个节点

BATCH_SIZE = 100 #一个训练batch的数据集，数字越小时，越接近随机梯度下降；数字越大时，训练越接近梯度下降。

LEARNING_RATE_BASE= 0.8 #基础学习率

LEARNING_RATE_DECAY= 0.99 #学习率的衰减率。

REGULARIZATION_RATE = 0.001 #描述模型复杂度的正则化项在损失函数中的系数

TRAINING_STEPS = 30000 #训练轮数

MOVING_AVERAGE_DECAY= 0.99 #滑动平均衰减率

#RelU激活函数的三层全连接神经网络，通过RelU激活函数实现了去线性化。在这个函数中也支持传入用于计算参数平均值的类，这样方便在测试的时候使用滑动平均模型

def inference(input_tensor,avg_class,weight1,biases1,weights2,biases2):
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weight1) + biases1)
        return  tf.matmul(layer1,weights2) + biases2

    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weight1))+ avg_class.average(biases1))
        return tf.matmul(layer1,avg_class.average(weights2)) + avg_class. average(biases2)

#训练模型的计算过程
def train(mnist):
    x  = tf.placeholder(tf.float32,[None,INPUT_DATA],name='x-input')
    y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='y-input')

    #生成隐藏层的参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_DATA,LAYER1_NODE],stddev = 0.1))
    biases1 = tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))

    #生成输出层的参数
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev =0.1))
    biases2 = tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))

    #计算在当前参数下的神经网络前向传播的结果。这里给出的是用于计算滑动平均的类为None,所以函数不会使用参数的滑动平均值。
    y = inference(x,None,weights1,biases1,weights2,biases2)

    #定义存储轮数的变量。这个变量不需要计算滑动平均的值，所以这里指定这个变量为不可训练的变量（trainable=False）
    global_step = tf.Variable(0,trainable = False)

    #给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类,给定训练轮数的变量可以加快训练早期变量的更新速度
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)

    #在所有代表神经网络参数的变量上使用滑动平均。其他辅助变量（比如global_step）就不需要了。tf.trainable_variables返回的就是图上集合
    #GraphKeys.TRAINABLE_VARIABLES中的元素。这个集合的元素就是所有没有指定trainable=False的参数

    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    average_y = inference(x,variable_averages,weights1,biases1,weights2,biases2)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))

    #计算在batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    #计算L2正则化损失函数。
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    #计算模型的正则化损失。一般只计算在神经网络边上权重的正则化损失，而不使用偏置项。
    regularization  = regularizer(weights1) + regularizer(weights2)

    #总损失等于交叉熵损失和正则化损失的和。
    loss = cross_entropy_mean + regularization

    #设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,mnist.train.num_examples/BATCH_SIZE
                                               ,LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

    with tf.control_dependencies([train_step,variable_averages_op]):
        train_op = tf.no_op(name='train')

        correct_prediction = tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        #初始化回话并开始训练过程。
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        validate_feed = {x:mnist.validation.images,y_:mnist.validation.labels}
        test_feed = {x:mnist.test.images,y_:mnist.test.labels}

        for i in range(TRAINING_STEPS):
            if i%1000 ==0:
                validate_acc = sess.run(accuracy,feed_dict=validate_feed)
                print ("After %d training step(s),validation accuracy using avarage mode is %g"%(i,validate_acc))

            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})

        test_acc = sess.run(accuracy,feed_dict=test_feed)
        print ("After %d training step(s),test accuracy using average mode is %g" %(TRAINING_STEPS,test_acc))

#主程序入口
def main(argv=None):
    mnist =input_data.read_data_sets("../Mnist_data",one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()
