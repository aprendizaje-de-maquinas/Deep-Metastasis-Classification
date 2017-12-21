# Copyright Jonathan Booher 2017. All rights reserverd

import tensorflow as tf


#from networkv2 import defineNetwork
#from network import defineNetwork
#from networkv3 import defineNetwork
from networkv4 import defineNetwork


from xqueue import TensorflowQueue
from utils import getInput
from utils import getInputTest
import time
import threading
from math import ceil

batch_size = 10
num_epochs = 1500

downscalesize = 224
numRegions= 5
numclasses =2

dropout = 0.3

test_set_size = 100

train_data = 'train-labels.csv'
train_data_phase_2 = 'multiclass-train-positives-bal2.csv'
train_data_phase_3 = 'multiclass-train-positives.csv'
train_data_phase_4 = 'multiclass-train.csv'


test_data = 'test-labels.csv'
log_dir = 'tensorboard-training-logs/500-resnet-binary-2'



config = tf.ConfigProto()
config.gpu_options.allow_growth = True


#required to run tf
with tf.Session(config = config) as sess:


    # shapes note the last dim is RGB
    inputShape = [ numRegions , downscalesize , downscalesize , 3 ]
    lableShape = [numclasses]

    #to scale the patches dow to manageable size
    warpDim =  (downscalesize,downscalesize)

    #fore threading
    coord = tf.train.Coordinator()

    #holds the inputs. Custom
    reader = TensorflowQueue( train_data , sess , getInput , numRegions , inputShape , lableShape , coord , warpDim=warpDim , batchSize = batch_size )

    
    train_step, loss , met_test , keep_prob, global_step = defineNetwork(sess , numclasses , reader , inputShape , lableShape )
    
    times = []

    #init all the varaibles
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    #for the summaries
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter( log_dir , sess.graph)
    saver = tf.train.Saver(max_to_keep=None)

    #saver.restore( sess , 'saves/resnet-mutli-bin_phase_1.ckpt' )
    
    threads = reader.start_threads(sess)
    
    #train loop
    for i in range( num_epochs ):

        start = time.time()

        #run one iteration of the optimizer
        _, l = sess.run( [train_step , loss  ] , feed_dict={ keep_prob:1.0-dropout , global_step: i} )
        times.append( time.time() - start )
        print ( 'Up to iter   '+str(i) +  '    average time is   ' + str( sum( times )/ len(times ) ) )

        writer.add_summary( l , global_step = i )

    saver.save( sess , 'saves/resnet-mutli-bin_phase_1.ckpt' ) 
    # training done join the threads
    coord.request_stop()
    print("stop requested.")
    for thread in threads:
        #sometimes the thread was blocking indefinetly, so set timeout
        thread.join(7)
    
    '''
    # new coordinator for testing
    coord = tf.train.Coordinator()

    #new queue -- easiest.
    reader.queue = tf.FIFOQueue(30, dtypes=[tf.float32 , tf.float32 ] , shapes=[inputShape,lableShape])

    reader.coord = coord
    # change the input
    reader.f_name = train_data_phase_2

    reader.threads = []
    threads = reader.start_threads( sess )
    print ( 'Beginning Phase 2' )
    #train loop
    for i in range( num_epochs ):

        start = time.time()

        #run one iteration of the optimizer
        _, l = sess.run( [train_step , loss  ] , feed_dict={ keep_prob:1.0-dropout , global_step: i+num_epochs} )
        times.append( time.time() - start )
        print ( 'Up to iter   '+str(i) +  '    average time is   ' + str( sum( times )/ len(times ) ) )

        writer.add_summary( l , global_step = i+num_epochs )
    

    saver.save( sess , 'saves/resnet-mutli-pos_phase_2.ckpt' )
    
    # training done join the threads
    coord.request_stop()
    print("stop requested.")
    for thread in threads:
        #sometimes the thread was blocking indefinetly, so set timeout
        thread.join(7)

    
    # new coordinator for testing
    coord = tf.train.Coordinator()

    #new queue -- easiest.
    reader.queue = tf.FIFOQueue(30, dtypes=[tf.float32 , tf.float32 ] , shapes=[inputShape,lableShape])

    reader.coord = coord
    # change the input
    reader.f_name = train_data_phase_3

    reader.threads = []
    threads = reader.start_threads( sess )
    print ( 'Beginning Phase 3' )
    #train loop
    for i in range( num_epochs ):

        start = time.time()

        #run one iteration of the optimizer
        _, l = sess.run( [train_step , loss  ] , feed_dict={ keep_prob:1.0-dropout , global_step: i+num_epochs} )
        times.append( time.time() - start )
        print ( 'Up to iter   '+str(i) +  '    average time is   ' + str( sum( times )/ len(times ) ) )

        writer.add_summary( l , global_step = i+2*num_epochs )
    

    saver.save( sess , 'saves/resnet-mutli-pos_phase_3.ckpt' ) 
    
    # training done join the threads
    coord.request_stop()
    print("stop requested.")
    for thread in threads:
        #sometimes the thread was blocking indefinetly, so set timeout
        thread.join( sum( times )/ len(times )  )


    # new coordinator for testing
    coord = tf.train.Coordinator()

    #new queue -- easiest.
    reader.queue = tf.FIFOQueue(30, dtypes=[tf.float32 , tf.float32 ] , shapes=[inputShape,lableShape])

    reader.coord = coord
    # change the input
    reader.f_name = train_data_phase_4

    reader.threads = []
    threads = reader.start_threads( sess )
    print ( 'Beginning Phase 4' )
    #train loop
    for i in range( num_epochs ):

        start = time.time()

        #run one iteration of the optimizer
        _, l = sess.run( [train_step , loss  ] , feed_dict={ keep_prob:1.0-dropout , global_step: i+num_epochs} )
        times.append( time.time() - start )
        print ( 'Up to iter   '+str(i) +  '    average time is   ' + str( sum( times )/ len(times ) ) )

        writer.add_summary( l , global_step = i+3*num_epochs )
    

    saver.save( sess , 'saves/scratch_phase_4.ckpt' ) 
    
    
    # training done join the threads
    coord.request_stop()
    print("stop requested.")
    for thread in threads:
        #sometimes the thread was blocking indefinetly, so set timeout
        thread.join( sum( times )/ len(times )  )
    '''
    
    # new coordinator for testing
    coord = tf.train.Coordinator()

    #new queue -- easiest.
    reader.queue = tf.FIFOQueue(30, dtypes=[tf.float32 , tf.float32 ] , shapes=[inputShape,lableShape])

    reader.coord = coord
    # change the input generator
    reader.inputGenerator= getInputTest
    reader.f_name = test_data

    reader.threads = []
    threads = reader.start_threads( sess )

    # itera over all test data
    for j in range( 0  ,  ceil( test_set_size / batch_size ) ):

        _ = sess.run( met_test , feed_dict={keep_prob:1.0} )
        summ = sess.run( merged , feed_dict={keep_prob:1.0 , global_step: 2*num_epochs+ j } )
        writer.add_summary( summ ,global_step=j+4*num_epochs )

    # join threads again
    coord.request_stop()
    print("stop requested.")
    for thread in threads:
        thread.join( sum( times )/ len(times ) )
    
    


    

