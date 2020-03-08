# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import math

N_HIDDEN_1 = 64
N_HIDDEN_2 = 64
TAU = 1e-3

ALPHA = 1e-2
EPSILON = 1e-6

class Q_Network:
    def __init__(self, seed, dim_obs, dim_act, num_act, lr_A, lr_V):
        self.seed, self.lr_A, self.lr_V = seed, lr_A, lr_V
        self.num_act = num_act
        self.dim_state, self.dim_action = dim_obs, dim_act
        
        tf.set_random_seed(seed)
        self.g = tf.Graph()
        with self.g.as_default():
            self.create_placeholder()
            self.create_anet()
            self.create_vnet()
            self.create_optimizer()
            self.init_session()
            self.saver = tf.train.Saver()
            
    def create_placeholder(self):
        self.state_in = tf.placeholder("float",[None, self.dim_state])
        self.action_in = tf.placeholder("float",[None, self.num_act])
        self.td_in = tf.placeholder("float",[None])
        self.v_value_in = tf.placeholder("float",[None,1])
    
    def create_anet(self):
        tf.set_random_seed(self.seed)
        with tf.variable_scope('a_network'):
            h1 = tf.layers.dense(self.state_in, N_HIDDEN_1, tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(seed = self.seed), name='H1')
            h2 = tf.layers.dense(h1, N_HIDDEN_2, tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(seed = self.seed), name='H2')
            self.a_probs = tf.layers.dense(h2, self.num_act, tf.nn.softmax,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(seed = self.seed), name='critic')
            
        self.weights_a = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='a_network')
        
    def create_vnet(self):
        tf.set_random_seed(self.seed)
        with tf.variable_scope('v_network'):
            h1 = tf.layers.dense(self.state_in, N_HIDDEN_1, tf.nn.relu,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='H1')
            h2 = tf.layers.dense(h1, N_HIDDEN_2, tf.nn.relu,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='H2')
            self.v_predict = tf.layers.dense(h2, 1,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='critic')
            
        self.weights_v = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='v_network')
        
    def create_optimizer(self):
        self.a_loss = -tf.reduce_sum(tf.log(tf.reduce_sum(self.a_probs*self.action_in, axis=1))*self.td_in) + ALPHA*tf.reduce_sum(self.a_probs*tf.log(self.a_probs+EPSILON))
        self.a_gradients = tf.gradients(self.a_loss, self.weights_a)
        self.a_optimizer = tf.train.AdamOptimizer(self.lr_A).apply_gradients(zip(self.a_gradients, self.weights_a))
        
        self.v_error = self.v_predict-self.v_value_in
        self.v_loss = tf.reduce_mean(tf.pow(self.v_error,2))
        self.v_gradients = tf.gradients(self.v_loss, self.weights_v)
        self.v_optimizer = tf.train.AdamOptimizer(self.lr_V).apply_gradients(zip(self.v_gradients, self.weights_v))
        
    def init_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config,graph=self.g)
        self.sess.run(tf.global_variables_initializer())
    
    def get_a(self, state_t):
        probs = self.sess.run(self.a_probs, feed_dict={self.state_in: np.reshape(state_t, [1,-1])})
        return probs
    
    def get_v(self, state_t):
        vs = self.sess.run(self.v_predict, feed_dict={self.state_in:state_t})
        return vs
    
    def get_a_gradients(self, state_t_batch, action_t_batch, td_batch):
        return self.sess.run([self.a_loss, self.a_gradients], feed_dict={self.state_in:state_t_batch, self.action_in:action_t_batch, self.td_in:td_batch})
        
    def get_v_gradients(self, state_t_batch, v_batch):
        v_batch = np.reshape(v_batch, [-1,1])
        return self.sess.run([self.v_loss, self.v_gradients], feed_dict={self.state_in:state_t_batch, self.v_value_in:v_batch})
    
    def apply_a_gradients(self, a_grad_batch):
        return self.sess.run(self.a_optimizer, feed_dict={i: d for i, d in zip(self.a_gradients, a_grad_batch)})
    
    def apply_v_gradients(self, v_grad_batch):
        return self.sess.run(self.v_optimizer, feed_dict={i: d for i, d in zip(self.v_gradients, v_grad_batch)})
    
    def save_network(self, _name):
        self.saver.save(self.sess, "./weights/"+_name+".ckpt")
        #print("* Successfully save the networks.")
        
    def load_network(self, _name):
        self.saver.restore(self.sess, "./weights/"+_name+".ckpt")
        #print("* Successfully load the networks.")
