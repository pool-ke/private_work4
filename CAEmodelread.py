#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 08:51:39 2017

@author: root
"""

import numpy as np
from PIL import Image
import skimage.morphology as sm
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage import filters,io, measure, color
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

inputs_ = tf.placeholder(tf.float32,(None, 695, 161, 1), name = 'inputs_')
targets_ = tf.placeholder(tf.float32,(None, 695, 161, 1), name = 'targets_')
#hidden layer

conv1_p = tf.layers.conv2d(inputs_,32, (5,5), padding = 'same', activation = tf.nn.relu, name = 'conv1_p')
conv1_d = tf.layers.max_pooling2d(conv1_p, (5,5),(5,5), padding = 'same', name = 'conv1_d')

conv2_p = tf.layers.conv2d(conv1_d,32, (5,5), padding = 'same', activation = tf.nn.relu, name = 'conv2_p')
conv2_d = tf.layers.max_pooling2d(conv2_p, (5,5),(5,5), padding = 'same', name = 'conv2_d')

conv3_p = tf.layers.conv2d(conv2_d,32, (5,5), padding = 'same', activation = tf.nn.relu, name = 'conv3_p')
conv3_d = tf.layers.max_pooling2d(conv3_p, (2,2),(2,2), padding = 'same', name = 'conv3_d')


in_full_connect = tf.reshape(conv3_d,[-1,14*4*32], name='in_full_connect')#upfold the tensor


conv4_n = tf.image.resize_nearest_neighbor(conv3_d,(28,7), name = 'conv4_n')
conv4_d = tf.layers.conv2d(conv4_n, 32, (5,5),padding = 'same',activation = tf.nn.relu, name = 'conv4_d')

conv5_n = tf.image.resize_nearest_neighbor(conv4_d,(139,33), name = 'conv5_n')
conv5_d = tf.layers.conv2d(conv5_n, 32, (5,5),padding = 'same',activation = tf.nn.relu, name = 'conv5_d')

conv6_n = tf.image.resize_nearest_neighbor(conv5_d,(695, 161),name = 'conv6_n')
conv6_d = tf.layers.conv2d(conv6_n,32, (5,5),padding = 'same',activation = tf.nn.relu, name = 'conv6_d')

logits_ = tf.layers.conv2d(conv6_d, 1, (5,5), padding = 'same', activation = None, name = 'logits_')
outputs_ = tf.nn.sigmoid(logits_, name = 'outputs_')

#loss function
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=inputs_,logits=outputs_, name='loss')
cost = tf.reduce_mean(loss, name='cost')

#optimal function
learning_rate = 0.001
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)



def test_model(image_input):
#    img2, label2 = read_and_decode("/home/huawei/myfile/code_python/Feng/LOGO_train_More_15.tfrecords",flag = 4)#/home/huawei/myfile/code_python/Feng/tensorflow/LOGO_train_NG_695.tfrecords
    saver2 = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess = sess, coord = coord)
#        val2, _ = sess.run([img2, label2])
#        val2 = val2/255.0
        saver2.restore(sess,"./wuquanlianjie/model.ckpt")#./Model/model.ckpt
    
        encode_feature = sess.run(in_full_connect, feed_dict = {inputs_: image_input})
        coord.request_stop()
        coord.join(threads)
        return encode_feature