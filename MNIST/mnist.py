#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os

mnist = input_data.read_data_sets("../Mnist_data",one_hot = True)

print "Trainning data size:",mnist.train.num_examples
print "Validation data size:",mnist.validation.num_examples
print "Test data size:",mnist.test.num_examples
print "Example training data:",mnist.train.images[0]
print "Example trainning data label:",mnist.train.labels[0]



