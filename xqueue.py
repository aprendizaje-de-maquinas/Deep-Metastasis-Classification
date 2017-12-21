

import tensorflow as tf
import numpy as np
import os
from utils import searchAndUpsample
import csv

 
from matplotlib import pyplot as plt
import threading
import time

#class for async loading of images into a FiFo queue for tf
class TensorflowQueue(object):
    def __init__(self, f_name , sess , inputGenerator , numRegions , inputShape , outputShape , coord, warpDim , n_threads=4 , batchSize=2 , max_queue_size=30, wait_time=0.01):

        self.numRegions = numRegions
        self.warpDim = warpDim
        self.getInput = inputGenerator
        self.n_threads = n_threads

        # to prevent file io problems we have a copy of all of these for each thread
        inputGenerator.mr_image = [ None ]* self.n_threads
        inputGenerator.reader = [ None ]* self.n_threads
        inputGenerator.name = [ None ]* self.n_threads

        
        self.wait_time = wait_time
        self.max_queue_size = max_queue_size
        self.queue = tf.FIFOQueue(max_queue_size, dtypes=[tf.float32 , tf.float32 ] , shapes=[inputShape,outputShape])
        self.queue_size = self.queue.size()
        self.threads = []
        self.coord = coord
        self.inpt = tf.placeholder(dtype=tf.float32, shape=inputShape)
        self.lbl = tf.placeholder(dtype=tf.float32, shape=outputShape)
        self.batchSize = batchSize
        self.enqueue = self.queue.enqueue([self.inpt , self.lbl])
        self.f_name = f_name
        self.sess = sess

    def dequeue(self):
        # dequeue queue size unless size of queue is less than batch size in that case dequeue size of
        #  queue. Note in the special case that there is nothing in the queue, we set to batchsize as
        #  that occurs when tf is initing everything
        q_size = self.queue_size.eval( session=self.sess)
        if q_size == 0:
            q_size = self.batchSize
        return self.queue.dequeue_many( min( self.batchSize , q_size  ) )

    def thread_main(self, i , sess):
        # this runs while the thread have not been joined
        stop = False
        while not stop:
            # get input from pipeline
            data,lbl = self.getInput( self.f_name , i , self.numRegions , self.warpDim)
            while self.queue_size.eval(session=sess) == self.max_queue_size:
                if self.coord.should_stop():
                    break
                time.sleep(self.wait_time)
            if self.coord.should_stop():
                # we should stop
                stop = True
                print("Enqueue thread receives stop request.")
                break
            sess.run(self.enqueue, feed_dict={self.inpt: data , self.lbl: lbl})

    def start_threads(self, sess ):
        # starts allt he thread running
        self.sess = sess
        for i in range(self.n_threads):
            thread = threading.Thread(target=self.thread_main, args=(i , sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads
       

# example 
if __name__ == '__main__':

    from utils import getInput

    
    batch_size = 3
    
    coord = tf.train.Coordinator()


    reader = TensorflowQueue(getInput , 5 , [5,512,512,3] , [2] , coord , (512,512) , batchSize = batch_size)
    net_input, lbl = reader.dequeue()


    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    init = tf.global_variables_initializer()
    sess.run(init)

    threads = reader.start_threads(sess)


    net = net_input
    l = lbl
    

    
    queue_size = reader.queue_size
    for step in range(1000):
        print('step    ' + str(step) +'    size queue =', queue_size.eval(session=sess))
        
        start = time.time()
        [img,ll] = sess.run([net , l])
        print ( time.time() - start )
        
        time.sleep(2)

    coord.request_stop()
    print("stop requested.")
    for thread in threads:
        thread.join()
        
    
