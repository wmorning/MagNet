'''
A class to hold all of our networks
'''

import tensorflow as tf
slim = tf.contrib.slim
import time
import numpy as np
from get_data import DataProcessor
from DenseNet import DenseNet

# =================================================================================================

class MagNet(object):
    '''
    '''
    
    def __init__(self,numpix_side,pixel_size,m,datadir,bad_files_list = None, bad_test_files_list = None, downsample=1):
        # initialize data processor
        self.numpix_side = numpix_side
        self.numpix_out = numpix_side / downsample
        assert numpix_side % downsample ==0
        self.DataProcessor = DataProcessor(datadir,m,numpix_side,pixel_size,bad_files_list = bad_files_list, bad_test_files_list = bad_test_files_list,downsample=downsample)
        
        # initialize placeholders
        self.initialize_placeholders()
        
        return
        
    def initialize_placeholders(self):
        '''
        initialize the placeholders (x,y) and their reshaped tensors (x_image,y_image)
        '''
        self.x = tf.placeholder(tf.float32,[None,self.numpix_side**2])
        self.x_image = tf.reshape(self.x,[-1,self.numpix_side,self.numpix_side,1]) 
        self.y_= tf.placeholder(tf.float32,[None,self.numpix_out**2])
        self.y_image = tf.reshape(self.y_,[-1,self.numpix_out,self.numpix_out,1])
        return
        
    def Choose_Network(self,network_name,TRANSFORM=False,densenet_arch='conv'):
        '''
        Pick the network.  for now, there is only one 
        valid choice, which is the AutoEncoder.  We can add 
        a spatial transformer to the front end of it.
        
        When we add more networks, the only requirement is that they take in x_image, and 
        output y_conv and cost.
        '''
        if TRANSFORM:
            self.TRANSFORM = True
            self.x_image , self.net_x, self.net_y = model_transformer(self.x_image , scope='transformer')
        else:
            self.TRANSFORM = False

        if network_name == 'DenseNet':
            self.y_conv, self.is_training, self.keep_prob = DenseNet(self.x_image,numpix_out = self.numpix_out,arch=densenet_arch)
            self.cost = Cost_AutoEncoder(self.y_conv,self.y_image)
            self.network_variable_scope = 'DenseNet'

        if network_name == 'Ensai_Autoencoder':
            self.y_conv = Ensai_AutoEncoder(self.x_image,scope='AutoEncoder')
            self.cost   = Cost_AutoEncoder(self.y_conv,self.y_image)
            self.network_variable_scope = 'AutoEncoder'
            
    def Initialize_Session(self,restore_file=None):
        
        self.saver    = tf.train.Saver(slim.get_variables(scope=self.network_variable_scope))
        self.restorer = tf.train.Saver(slim.get_variables(scope=self.network_variable_scope))
        
        if self.TRANSFORM:
            self.transformer_saver = tf.train.Saver(slim.get_variables(scope='transformer'))
        
        self.train_step = tf.train.AdamOptimizer(5.0e-7).minimize(self.cost)
        self.session    = tf.Session()
        self.session.run(tf.global_variables_initializer())
        
        if restore_file is not None:
            self.restorer.restore(self.session,restore_file+'reconstructor_network_checkpoint.ckpt')
            if self.TRANSFORM:
                self.transformer_saver.restore(self.session,restore_file_prefix+'transformer_network_checkpoint.ckpt')
    
    def Train(self,Nsteps,save_every=10**8,save_file_prefix=''):
        '''
        Run m training steps.  
        '''
        
        # save cost in array, just in case we want to plot it
        self.current_cost = np.zeros(Nsteps)
        
        for n in range(Nsteps):
            
            tstart = time.time()
            self.DataProcessor.load_data_batch(100000,'train')
            _ , self.current_cost[n] = self.session.run([self.train_step,self.cost],feed_dict = {self.x:self.DataProcessor.X,self.y_:self.DataProcessor.Y,self.is_training:True,self.keep_prob:1.0})
            
            print n , self.current_cost[n] , time.time()-tstart
            
            if (n % save_every ==0) & (n !=0):
                self.saver.save(self.session,save_file_prefix+'reconstructor_network_checkpoint.ckpt')
                if self.TRANSFORM:
                    self.transformer_saver.save(self.session,save_file_prefix+'transformer_network_checkpoint.ckpt')
            
            
            
# ------------------------------------------- Networks ------------------------------------------- #

trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)
def model_transformer(x_image , scope="transformer", reuse=None):
        with tf.variable_scope(scope):
                with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=tf.nn.relu):
                        net = slim.conv2d(x_image, 64, [11, 11], 4, padding='VALID', scope='conv1')
                        net = slim.max_pool2d(net, [2, 2], scope='pool1')
                        net = slim.conv2d(net, 256, [5, 5], padding='VALID', scope='conv2')
                        net = slim.max_pool2d(net, [2, 2], scope='pool2')
                        net = slim.conv2d(net, 512, [3, 3], scope='conv3')
                        net = slim.conv2d(net, 1024, [3, 3], scope='conv4')
                        net = slim.conv2d(net, 1024, [3, 3], scope='conv5')
                        net = slim.max_pool2d(net, [2, 2], scope='pool5')
                        with slim.arg_scope([slim.conv2d], weights_initializer=trunc_normal(0.005), biases_initializer=tf.constant_initializer(0.1)):
                                net = slim.conv2d(net, 3072, [2, 2], padding='VALID', scope='fc6')
                                net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                                net = slim.conv2d(net, 5, [1, 1], activation_fn=None, normalizer_fn=None,  biases_initializer=tf.zeros_initializer(), scope='fc8')
                                net = slim.flatten(net, scope='Flatten')
                                net = slim.fully_connected(net,  5  , activation_fn = None ,  scope='FC1')
                net_x = slim.fully_connected(net, 1 , activation_fn=None , scope='TF_predict_x')
                net_y = slim.fully_connected(net, 1 , activation_fn=None , scope='TF_predict_y')
		zero_col = net_x * 0
		one_col = net_x * 0 + 1
		transformation_tensor = tf.concat( axis = 1 , values = [one_col, zero_col , 1.0 * net_x / ((192*0.04)/2), zero_col , one_col , 1.0 * net_y / ((192*0.04)/2) ] )
		x_image = transformer(x_image, transformation_tensor , (numpix_side, numpix_side) )
		x_image = tf.reshape(x_image , [-1,numpix_side,numpix_side,1] )
	return x_image, net_x, net_y
        
        
def Ensai_AutoEncoder( inputs , scope="AutoEncoder", reuse=None):
    with tf.variable_scope(scope):

        scale = 0.7
        with slim.arg_scope([slim.fully_connected, slim.layers.conv2d_transpose, slim.conv2d], activation_fn=tf.nn.relu ):


            net_c0 = slim.conv2d( inputs , 32, [3, 3], stride = 1 , scope='conv0')
            net_c1 = slim.conv2d(net_c0, 32, [3, 3], stride = 2 , scope='conv1')
            net_c2 = slim.conv2d(net_c1, 64, [3, 3], stride = 2 , scope='conv2')
            net_c3 = slim.conv2d(net_c2, 128, [5, 5], scope='conv3')
            net_c4 = slim.conv2d(net_c3, 64, [3, 3], stride = 1 , scope='conv4')

            
            # I commented out the bizarre residual connections that Y put in, since I don't see their purpose.
            # (They do not appear in autoencoder literature)
            net_d4 = slim.layers.conv2d_transpose( net_c4 , 64 , 3 , stride = 1, scope='deconvt4')
            #net_d4 = net_c4 + scale * net_d4

            net_d3 = slim.layers.conv2d_transpose( net_d4 , 128 , 5 , stride = 1, scope='deconvt3')
            #net_d3 = net_c3 + scale * net_d3

            net_d2 = slim.layers.conv2d_transpose( net_d3 , 64 , 3 , stride = 1, scope='deconvt2')
            #net_d2 = net_c2 + scale * net_d2

            net_d1 = slim.layers.conv2d_transpose( net_d2 , 32 , 3 , stride = 2, scope='deconvt1')
            #net_d1 = net_c1 + scale * net_d1

            net_d0 = slim.layers.conv2d_transpose( net_d1 , 32 , 3 , stride = 2 , scope='deconvt0')
            #net_d0 = net_c0 + scale * net_d0

            net = slim.layers.conv2d_transpose( net_d0 , 1 , 3 , activation_fn=None, stride = 1, scope='deconvt6')
            #split0, split1 = tf.split(inputs , 2 , 3)
            #net = split0 + scale * net

            return net
            
def Cost_AutoEncoder(y_conv,y_image):
    MeanSquareCost = tf.reduce_mean(tf.pow(y_conv-y_image ,2.) )
    return MeanSquareCost
