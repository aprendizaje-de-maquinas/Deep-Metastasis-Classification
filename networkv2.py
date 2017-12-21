# Copyright Jonathan Booher 2017. All rights reserverd

import tensorflow as tf
import numpy as np
import os
from xqueue import TensorflowQueue
import re

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
WEIGHT_DECAY_FACTOR = 0.004 / 5. # 3500 -> 2.8

TOWER_NAME = 'tower'
DEFAULT_PADDING = 'SAME'


# THIS IS AN IMPLEMENTATION OF ALEXNET
# EVERYTHING NOT MARKed SHOULD BE PRETTY SELF EXPLANTORY

def _activation_summary(x):

    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))



def _variable_on_cpu(name, shape, initializer):

    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, wd):

    var = _variable_on_cpu(name, shape,
                           initializer=tf.truncated_normal_initializer(stddev=0.001))
    if wd:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

def _conv(name, in_ ,ksize, strides=[1,1,1,1], padding=DEFAULT_PADDING, group=1, reuse=False , skip=None):
    
    n_kern = ksize[3]
    convolve = lambda i, k: tf.nn.conv2d(i, k, strides, padding=padding)

    with tf.variable_scope(name, reuse=reuse) as scope:
        if group == 1:
            kernel = _variable_with_weight_decay('weights', shape=ksize, wd=0.0)
            conv = convolve(in_, kernel)
        else:
            ksize[2] /= group
            kernel = _variable_with_weight_decay('weights', shape=ksize, wd=0.0)
            input_groups = tf.split( in_ , group , 3 )
            kernel_groups = tf.split(kernel, group, 3)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            # Concatenate the groups
            conv = tf.concat(output_groups, 3)

        biases = _variable_on_cpu('biases', [n_kern], tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
        if skip != None:
            conv = tf.nn.relu( conv+ skip , name = scope.name )
        else:
            conv = tf.nn.relu(conv, name=scope.name)

    print (name, conv.get_shape().as_list())
    return conv
def _maxpool(name, in_, ksize, strides, padding=DEFAULT_PADDING):
    pool = tf.nn.max_pool(in_, ksize=ksize, strides=strides,
                          padding=padding, name=name)

    print (name, pool.get_shape().as_list())
    return pool
def _fc(name, in_, outsize, dropout=1.0, reuse=False):
    with tf.variable_scope(name, reuse=reuse) as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        
        insize = in_.get_shape().as_list()[-1]
        weights = _variable_with_weight_decay('weights', shape=[insize, outsize], wd=0.004)
        biases = _variable_on_cpu('biases', [outsize], tf.constant_initializer(0.0))
        fc = tf.nn.relu(tf.matmul(in_, weights) + biases, name=scope.name)
        fc = tf.nn.dropout(fc, dropout)

        _activation_summary(fc)

    
    #print (name, fc.get_shape().as_list())
    return fc

# we take the activations from each of the parallel instances oft he network and
#   add them together for the pooling
def _patch_pool(patch_features, name):

    pp = tf.expand_dims(patch_features[0], 0) # eg. [100] -> [1, 100]
    for p in patch_features[1:]:
        p = tf.expand_dims(p, 0)
        pp = tf.concat([pp, p], 0)
    #print 'vp before reducing:', vp.get_shape().as_list()
    pp = tf.reduce_sum(pp, [0], name=name)
    return pp
    '''
    pp = patch_features[0] # eg. [100] -> [1, 100]

    for p in patch_features[1:]:
        #p = tf.expand_dims(p, 0)
        pp = tf.concat([pp, p], 1)
    '''
    return pp


def alex_pre_pool( patch , n_classes , keep_prob , reuse):
    
    conv1 = _conv('conv1', patch, [11, 11, 3, 96], [1, 4, 4, 1], 'VALID', reuse=reuse)
    lrn1 = tf.nn.local_response_normalization(conv1,
                                              alpha=1e-4,
                                              beta=0.75,
                                              depth_radius=2,
                                              bias=2.0)
    pool1 = _maxpool('pool1', lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
    conv2 = _conv('conv2', pool1, [5, 5, 96, 256], group=2, reuse=reuse)
    lrn2 = tf.nn.local_response_normalization(conv2,
                                              alpha=1e-4,
                                              beta=0.75,
                                              depth_radius=2,
                                              bias=2.0)
    pool2 = _maxpool('pool2', conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
    conv3 = _conv('conv3', pool2, [3, 3, 256, 384], reuse=reuse)
    conv4 = _conv('conv4', conv3, [3, 3, 384, 384], group=2, reuse=reuse)
    conv5 = _conv('conv5', conv4, [3, 3, 384, 256], group=2, reuse=reuse)

    pool5 = _maxpool('pool5', conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    dim = np.prod(pool5.get_shape().as_list()[1:])
    reshape = tf.reshape(pool5, [-1, dim])

    '''
    fc_dim = 2048
    fc6 = _fc('fc6', reshape , fc_dim, dropout=1.0 , reuse=reuse)
    fc7 = _fc('fc7', fc6, fc_dim, dropout=keep_prob , reuse = reuse)
    fc8 = _fc('fc8', fc7, n_classes , reuse = reuse)
    ''' 
    return  reshape
def alex_post_pool( img , n_classes , keep_prob ):
    fc_dim = 4096
    
    fc6 = _fc('fc6', img , fc_dim, dropout=keep_prob)
    fc7 = _fc('fc7', fc6, fc_dim, dropout=keep_prob)
    fc8 = _fc('fc8', fc7, n_classes)
    
    #fc7 = _fc('fc9', img, fc_dim, dropout=keep_prob)
    #fc8 = _fc('fc10', fc7, n_classes)
    return fc8

def net(images , n_classes):

    NUM_CLASSES = n_classes

    keep_prob = tf.placeholder(tf.float32)

    n_patches = images.get_shape().as_list()[1]
    images = tf.transpose( images , perm=[1,0,2,3,4] )

    patch_pool = []
    parameters = []

    #set up the n_patches parallel networks
    for i in range( n_patches ):
        patch_pool.append( alex_pre_pool( tf.gather( images , i ) , n_classes , keep_prob , (i != 0) ) )

    pool = _patch_pool( patch_pool , 'pach-pooling' )

    fc8  =  alex_post_pool( pool , NUM_CLASSES , keep_prob )
    
    return fc8, keep_prob , parameters


def _get_streaming_metrics(scope , prediction,label,num_classes):

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


def defineNetwork( n_classes , reader , inputShape , labelShape ):
    

    x, y_ = reader.dequeue()
    y_conv, keep_prob, params = net(x , n_classes)

    
    with tf.name_scope('loss'):
        #weights = tf.constant( [ 1.0 , 1.0 , 1.0 , 1.0 ] )
        #weights = tf.constant( [ 1.0 , 9.0 , 10.0 , 9.0 ] ) 
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits( labels=y_ , logits=y_conv )
        #cross_entropy = tf.nn.weighted_cross_entropy_with_logits(targets=y_, logits=y_conv , pos_weight=weights)
        cross_entropy = tf.scalar_mul( 10 , tf.reduce_mean(cross_entropy) )
    
        tf.add_to_collection('losses', cross_entropy)

        c = tf.add_n(tf.get_collection('losses'), name='total_loss')

    s = tf.summary.scalar( 'loss_' , c )

    with tf.name_scope('optimizer'):
        train_step = tf.train.MomentumOptimizer(10e-4 , 0.9).minimize(c)

    op_test , a , conf = _get_streaming_metrics( 'test' , tf.argmax(y_conv,1) , tf.argmax(y_,1) , n_classes ) 

    return train_step , s , op_test , keep_prob


if __name__ == '__main__':
    random = np.random.rand( 1,5,512,512,3)
    random = np.float32( random )

    with tf.Session() as sess:
        y , keep , params = net( random)
        
        print ( y.get_shape() )


