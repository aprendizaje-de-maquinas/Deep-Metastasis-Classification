
import tensorflow as tf
from nets import inception, resnet_v2
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def inference(input_tensor_batch, n_classes , reuse):
    #logits, endpoints = inception.inception_v4( input_tensor_batch , n_classes , scope='inception' , reuse=reuse )
    logits , argmax = resnet_v2.resnet_v2_101( input_tensor_batch , n_classes , reuse = reuse )
    return logits , argmax

def _view_pool(view_features, name):

    vp = tf.expand_dims(view_features[0], 0) # eg. [100] -> [1, 100]
    for v in view_features[1:]:
        v = tf.expand_dims(v, 0)
        vp = tf.concat([vp, v], 0)
    vp = tf.reduce_sum(vp, [0], name=name)
    return vp


def net(images , n_classes):


    keep_prob = tf.placeholder(tf.float32)

    n_patches = images.get_shape().as_list()[1]
    images = tf.transpose( images , perm=[1,0,2,3,4] )

    patch_pool = []
    parameters = []

    for i in range( n_patches ):
        logits , argmax = inference( tf.gather( images , i ) , n_classes , (i != 0) )
        patch_pool.append( logits )
        

    pool = _view_pool( patch_pool , 'pach-pooling' )
    fc8  =  pool
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

def defineNetwork( sess , n_classes , reader , inputShape , labelShape , epochs=500):
    

    x, y_ = reader.dequeue()
    y_conv, keep_prob, params = net(x , n_classes)

    
    with tf.name_scope('loss'):
        '''
        missclassification_weights = tf.constant( [ [ 1.0 , 1.0  ] ,
                                                    [ 5.0 , 1.0 ] ] )

        nb_cl = 2
        a , b = y_conv.get_shape()
        
        final_mask = tf.zeros( a , tf.float32   )
        y_pred_max = tf.reduce_max( y_conv , axis = 1 )
        y_pred_max = tf.reshape( y_pred_max , [int(a) , 1] )
        y_pred_max_mat = tf.cast( tf.equal( y_conv , y_pred_max ) , tf.float32 )
        for c_p , c_t in product( range( nb_cl ) , range( nb_cl ) ):
            final_mask += ( missclassification_weights[c_t,c_p]*y_pred_max_mat[:,c_p]*y_[:,c_t] )

        cross_entropy = tf.losses.softmax_cross_entropy( y_ , y_conv  ) * final_mask
        '''
        '''
        missclassification_weights = tf.constant( [ [ 0.0 , 1.0 , 1.0 , 10.0 ] ,
                                                    [ 10.0 , 0.0 , 1.0 , 10.0 ] ,
                                                    [ 10.0 , 1.0 , 0.0 , 10.0 ] ,
                                                    [ 20.0 , 1.0 , 1.0 , 0.0 ]  ]  )

        nb_cl = 4
        a , b = y_conv.get_shape()
        
        final_mask = tf.zeros( a , tf.float32   )
        y_pred_max = tf.reduce_max( y_conv , axis = 1 )
        y_pred_max = tf.reshape( y_pred_max , [int(a) , 1] )
        y_pred_max_mat = tf.cast( tf.equal( y_conv , y_pred_max ) , tf.float32 )
        for c_p , c_t in product( range( nb_cl ) , range( nb_cl ) ):
            final_mask += ( missclassification_weights[c_t,c_p]*y_pred_max_mat[:,c_p]*y_[:,c_t] )

        cross_entropy = tf.losses.softmax_cross_entropy( y_ , y_conv  ) * final_mask
        '''
        cross_entropy = tf.losses.softmax_cross_entropy( y_ , y_conv  )

        c = tf.reduce_mean( cross_entropy )
        
    s = tf.summary.scalar( 'loss_' , c )

    global_step = tf.Variable( 0 , trainable=False )
    boundaries = [ epochs  , 2* epochs , 3*epochs]
    values = [ 10e-4 , 10e-5 , 10e-6 , 10e-7 ] 
    values_ = [ 0.9 , 0.7 , 0.3 , 0.1 ] 

    with tf.name_scope('optimizer'):
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
        momentum = tf.train.piecewise_constant( global_step , boundaries , values_ )

        #train_step = tf.train.AdamOptimizer(10e-6).minimize(c)
        train_step = tf.train.MomentumOptimizer(learning_rate , momentum).minimize(c)

    op_test , a , conf = _get_streaming_metrics( 'test' , tf.argmax(y_conv,1) , tf.argmax(y_,1) , n_classes ) 

    return train_step , s , op_test , keep_prob , global_step


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

        print ( y_conv.get_shape() )
