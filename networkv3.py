

import numpy as np
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


BN_EPSILON = 0.001

# THIS IS THE SAME AS V2 BUT WITH RESNET.

def activation_summary(x):
    return
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):

    
    ## TODO: to allow different weight decay to fully connected layer and conv layer
    if is_fc_layer is True:
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.0)
    else:
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.)

    new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                    regularizer=regularizer)
    return new_variables

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

def _fc(name, in_, outsize, dropout=1.0, reuse=False):
    with tf.variable_scope(name, reuse=reuse) as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        
        insize = in_.get_shape().as_list()[-1]
        weights = _variable_with_weight_decay('weights', shape=[insize, outsize], wd=0.004)
        biases = _variable_on_cpu('biases', [outsize], tf.constant_initializer(0.0))
        fc = tf.nn.relu(tf.matmul(in_, weights) + biases, name=scope.name)
        fc = tf.nn.dropout(fc, dropout)

    
    #print (name, fc.get_shape().as_list())
    return fc


def output_layer(input_layer, num_labels):

    input_dim = input_layer.get_shape().as_list()[-1]
    fc_w = create_variables(name='fc_weights', shape=[input_dim, num_labels], is_fc_layer=True,
                            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    fc_b = create_variables(name='fc_bias', shape=[num_labels], initializer=tf.zeros_initializer())

    fc_h = tf.matmul(input_layer, fc_w) + fc_b
    return fc_h


def batch_normalization_layer(input_layer, dimension):

    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    beta = tf.get_variable('beta', dimension, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable('gamma', dimension, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))
    bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)

    return bn_layer


def conv_bn_relu_layer(input_layer, filter_shape, stride , reuse):

    out_channel = filter_shape[-1]
    filter = create_variables(name='conv', shape=filter_shape )

    conv_layer = tf.layers.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME' , reuse=reuse)
    bn_layer = batch_normalization_layer(conv_layer, out_channel)

    output = tf.nn.relu(bn_layer)
    return output


def bn_relu_conv_layer(input_layer, filter_shape, stride):

    in_channel = input_layer.get_shape().as_list()[-1]

    bn_layer = batch_normalization_layer(input_layer, in_channel)
    relu_layer = tf.nn.relu(bn_layer)

    filter = create_variables(name='conv', shape=filter_shape)
    conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    return conv_layer



def residual_block(input_layer, output_channel, first_block=False):

    input_channel = input_layer.get_shape().as_list()[-1]

    # When it's time to "shrink" the image size, we use stride = 2
    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')

    # The first conv layer of the first residual block does not need to be normalized and relu-ed.
    with tf.variable_scope('conv1_in_block'):
        if first_block:
            filter = create_variables(name='conv', shape=[3, 3, input_channel, output_channel])
            conv1 = tf.nn.conv2d(input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
        else:
            conv1 = bn_relu_conv_layer(input_layer, [3, 3, input_channel, output_channel], stride)

    with tf.variable_scope('conv2_in_block'):
        conv2 = bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel], 1)

    # When the channels of input layer and conv2 does not match, we add zero pads to increase the
    #  depth of input layers
    if increase_dim is True:
        pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='VALID')
        padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                     input_channel // 2]])
    else:
        padded_input = input_layer

    output = conv2 + padded_input
    return output


def inference(input_tensor_batch, n_classes , n, reuse):

    layers = []
    with tf.variable_scope('conv0', reuse=reuse):
        conv0 = conv_bn_relu_layer(input_tensor_batch, [3, 3, 3, 16], 1 , reuse)
        activation_summary(conv0)
        layers.append(conv0)

    for i in range(n):
        with tf.variable_scope('conv1_%d' %i, reuse=reuse):
            if i == 0:
                conv1 = residual_block(layers[-1], 16 , first_block=True )
            else:
                conv1 = residual_block(layers[-1], 16 )
            activation_summary(conv1)
            layers.append(conv1)

    for i in range(n):
        with tf.variable_scope('conv2_%d' %i, reuse=reuse):
            conv2 = residual_block(layers[-1], 32 )
            activation_summary(conv2)
            layers.append(conv2)

    for i in range(n):
        with tf.variable_scope('conv3_%d' %i, reuse=reuse):
            conv3 = residual_block(layers[-1], 64  )
            layers.append(conv3)
        #assert conv3.get_shape().as_list()[1:] == [8, 8, 64] , conv3.get_shape().as_list()[1:]

    with tf.variable_scope('fc', reuse=reuse):
        in_channel = layers[-1].get_shape().as_list()[-1]
        bn_layer = batch_normalization_layer(layers[-1], in_channel)
        relu_layer = tf.nn.relu(bn_layer)
        global_pool = tf.reduce_mean(relu_layer, [1, 2])

        assert global_pool.get_shape().as_list()[-1:] == [64]
        output = output_layer(global_pool, n_classes)
        layers.append(output)

    return layers[-1]

def _view_pool(view_features, name):

    vp = tf.expand_dims(view_features[0], 0) # eg. [100] -> [1, 100]
    for v in view_features[1:]:
        v = tf.expand_dims(v, 0)
        vp = tf.concat([vp, v], 0)

    #print 'vp before reducing:', vp.get_shape().as_list()
    vp = tf.reduce_sum(vp, [0], name=name)
    return vp
    '''
    vp = view_features[0] # eg. [100] -> [1, 100]

    for v in view_features[1:]:
        #v = tf.expand_dims(v, 0)
        vp = tf.concat([vp, v], 1)
    '''
    return vp
def alex_post_pool( img , n_classes , keep_prob ):
    fc_dim = 512
    
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

    TOWER_HEIGHT = 5

    for i in range( n_patches ):
        patch_pool.append( inference( tf.gather( images , i ) , n_classes , TOWER_HEIGHT , (i != 0) ) )

    pool = _view_pool( patch_pool , 'pach-pooling' )
    fc8  =  alex_post_pool( pool , NUM_CLASSES , keep_prob )
    #fc8 = pool 
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

from itertools import product
from functools import partial

def defineNetwork( sess , n_classes , reader , inputShape , labelShape ):
    

    x, y_ = reader.dequeue()
    y_conv, keep_prob, params = net(x , n_classes)

    
    with tf.name_scope('loss'):

        missclassification_weights = tf.constant( [ [ 0.0 , 1.0 , 1.0 , 10.0 ] ,
                                                    [ 10.0 , 0.0 , 1.0 , 10.0 ] ,
                                                    [ 10.0 , 1.0 , 0.0 , 10.0 ] ,
                                                    [ 10.0 , 1.0 , 1.0 , 0.0 ]  ]  )

        nb_cl = 4
        a , b = y_conv.get_shape()
        
        final_mask = tf.zeros( a , tf.float32   )
        y_pred_max = tf.reduce_max( y_conv , axis = 1 )
        y_pred_max = tf.reshape( y_pred_max , [int(a) , 1] )
        y_pred_max_mat = tf.cast( tf.equal( y_conv , y_pred_max ) , tf.float32 )
        for c_p , c_t in product( range( nb_cl ) , range( nb_cl ) ):
            final_mask += ( missclassification_weights[c_t,c_p]*y_pred_max_mat[:,c_p]*y_[:,c_t] )
        
        #conf = tf.contrib.metrics.confusion_matrix( tf.argmax( y_ , 1) , tf.argmax( y_conv , 1 ) , num_classes = n_classes )

        #c_weights = tf.reduce_sum( tf.multiply( y_ , weights ) , 1 )
        cross_entropy = tf.losses.softmax_cross_entropy( y_ , y_conv  ) * final_mask
        
        #weights = tf.reduce_sum ( tf.matmul( tf.cast( conf, tf.float32 ) , missclassification_weights ) )
        #cross_entropy = tf.nn.softmax_cross_entropy_with_logits( labels=y_ , logits= y_conv )
        #cross_entropy = tf.nn.weighted_cross_entropy_with_logits(targets=y_, logits=y_conv , pos_weight=weights)
        #cross_entropy = tf.scalar_mul( 10 , tf.reduce_mean(cross_entropy) )
        #cros_entropy = cross_entropy * weights

        c = tf.reduce_mean( cross_entropy )
        #tf.add_to_collection('losses', cross_entropy)

        #c = tf.add_n(tf.get_collection('losses'), name='total_loss')
        
    s = tf.summary.scalar( 'loss_' , c )

    with tf.name_scope('optimizer'):
        #train_step = tf.train.AdamOptimizer(10e-6).minimize(c)
        train_step = tf.train.MomentumOptimizer(10e-9 , 0.9).minimize(c)

    op_test , a , conf = _get_streaming_metrics( 'test' , tf.argmax(y_conv,1) , tf.argmax(y_,1) , n_classes ) 

    return train_step , s , op_test , keep_prob


if __name__ == '__main__':
    random = tf.constant ( np.random.rand( 4,5,512,512,3 ) , tf.float32 )
    yy = [ [ 1, 0 , 0 , 0 ] ,
           [ 0, 1 , 0 , 0 ] ,
           [ 0, 0 , 1 , 0 ] ,
           [ 0, 0 , 0 , 1 ] ] 
    y_ = tf.constant( yy , tf.float32 )  

    with tf.Session() as sess:
        y_conv , keep , params = net( random , 4 )

        sess.run(tf.global_variables_initializer() )

        missclassification_weights = np.array([ [ 0.0 , 1.0 , 1.0 , 1.0 ] ,
                                                [ 2.0 , 0.0 , 1.0 , 1.0 ] ,
                                                [ 3.0 , 3.0 , 0.0 , 1.0 ] ,
                                                [ 4.0 , 4.0 , 4.0 , 0.0 ]  ] )

        a , b = y_conv.get_shape()
        nb_cl = b

        
        final_mask = tf.zeros( a , tf.float32   )
        y_pred_max = tf.reduce_max( y_conv , axis = 1 )

        
        y_pred_max = tf.reshape( y_pred_max , [int(a) , 1] )

        y_pred_max_mat = tf.cast( tf.equal( y_conv , y_pred_max ) , tf.float32 ).eval( feed_dict={keep:1.0})
        yy = y_.eval()

        for c_p , c_t in product( range( nb_cl ) , range( nb_cl ) ):
            final_mask += ( missclassification_weights[c_t,c_p]*y_pred_max_mat[:,c_p]*y_[:,c_t] )
        

        cross_entropy = tf.losses.softmax_cross_entropy( y_ , y_conv  ) * final_mask

        c = tf.reduce_mean( cross_entropy )
        

