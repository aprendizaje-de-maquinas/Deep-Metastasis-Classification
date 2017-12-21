# Copyright Jonathan Booher 2017. All rights reserverd

import tensorflow as tf
import numpy as np
import os
from xqueue import TensorflowQueue

# minimizes the logging that tensorflow writes out to the terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# simple entwork. shouldbe pretty self explanatory
def deepnn(x , n_output):

    NUM_CLASSES = n_output

    CONV_ONE = 32
    CONV_TWO = 64
    FC_ONE = 512

    with tf.name_scope('conv1'):
        W_conv1 = weight_variable([ 1 , 3, 3, 3, CONV_ONE])
        b_conv1 = bias_variable([CONV_ONE])
        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)


    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2'):
        W_conv2 = weight_variable([ 1 , 3, 3, CONV_ONE,CONV_TWO])
        b_conv2 = bias_variable([CONV_TWO])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('fc1'):
        dim = int(h_pool2.get_shape()[1])*int(h_pool2.get_shape()[2])*int(h_pool2.get_shape()[3]) 

        W_fc1 = weight_variable([ dim*CONV_TWO ,  FC_ONE])
        b_fc1 = bias_variable([FC_ONE])
        
        h_pool2_flat = tf.reshape(h_pool2, [-1 , dim*CONV_TWO])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc2'):
        W_fc2 = weight_variable([FC_ONE, NUM_CLASSES])
        b_fc2 = bias_variable([NUM_CLASSES])


        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob


def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv3d(x, W, strides=[1 , 1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool3d(x, ksize=[1,1, 4, 4, 1],
                          strides=[1,1, 4, 4, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def _get_streaming_metrics(scope , prediction,label,num_classes=3):

    with tf.name_scope(scope):
        # the streaming accuracy (lookup and update tensors)
        accuracy,accuracy_update = tf.metrics.accuracy(label, prediction, 
                                               name='accuracy')
        # Compute a per-batch confusion
        batch_confusion = tf.contrib.metrics.confusion_matrix(label, prediction, num_classes=num_classes, 
                                             name='batch_confusion')
        # Create an accumulator variable to hold the counts
        confusion = tf.Variable( tf.zeros([num_classes,num_classes], 
                                          dtype=tf.int32 ),
                                 name='confusion' )

        auc, up = tf.metrics.auc( labels=label , predictions=prediction , name='auc_')

        
        # Create the update op for doing a "+=" accumulation on the batch
        confusion_update = confusion.assign( confusion + batch_confusion )
        # Cast counts to float so tf.summary.image renormalizes to [0,255]
        confusion_image = tf.reshape( tf.cast( confusion, tf.float32),
                                  [1, num_classes, num_classes, 1])
        # Combine streaming accuracy and confusion matrix updates in one op
        test_op = tf.group(accuracy_update, confusion_update , up)

        tf.summary.image('confusion',confusion_image)
        tf.summary.text('confusion-text' , tf.as_string( confusion  ) )
        
        tf.summary.scalar('accuracy',accuracy)
        tf.summary.scalar('auc' , auc )

    return test_op,accuracy,confusion




def defineNetwork( num_output , reader , inputShape , lableShape ,  batch_size=3 ):
    
    x, y_ = reader.dequeue()
    y_conv, keep_prob = deepnn(x , num_output)

    
    with tf.name_scope('loss'):
        #weights = tf.constant( [ 1.0 , 1.0 , 1.0 ] ) 

        #weights = tf.constant( [ 1.652 , 12.903 , 7.407 , 5.479 ] ) 
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits( labels=y_ , logits=y_conv )
        #cross_entropy = tf.nn.weighted_cross_entropy_with_logits(targets=y_, logits=y_conv , pos_weight=weights)
        cross_entropy = tf.reduce_mean(cross_entropy)

    s = tf.summary.scalar( 'loss_' , cross_entropy ) 

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    tf.summary.scalar( 'zero-one-accuracy' , accuracy )


    op_test , a , conf = _get_streaming_metrics( 'test' , tf.argmax(y_conv,1) , tf.argmax(y_,1) ) 

    return train_step, s , op_test , keep_prob


    
