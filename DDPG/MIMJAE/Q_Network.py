# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import math
from utils import *

TAU = 0.001
#N_HIDDEN_1, N_HIDDEN_2 = 64, 64
N_HIDDEN_1, N_HIDDEN_2 = 400, 300

class Q_Network:
    def __init__(self, seed, dim_obs, dim_act, L_act, batch_size, lr_A, lr_Q):
        self.seed = seed
        tf.set_random_seed(seed)
        self.dim_state, self.dim_action, self.action_limit = dim_obs, dim_act, L_act
        self.batch_size, self.lr_A, self.lr_Q = batch_size, lr_A, lr_Q
        
        self.g = tf.Graph()
        with self.g.as_default():
            self.create_placeholder()
            self.create_actor()
            self.create_critic()
            self.create_optimizer()
            self.create_update_operation()
            self.init_session()
            self.saver = tf.train.Saver()
            
    def create_placeholder(self):
        self.state_in = tf.placeholder("float",[None, self.dim_state])
        self.action_in = tf.placeholder("float",[None, self.dim_action])
        
        self.scale = tf.placeholder("float")
        self.q_value_in = tf.placeholder("float",[None,1])
        self.q_gradient_in = tf.placeholder("float",[None,self.dim_action])
    
    def create_actor(self):
        tf.set_random_seed(self.seed)
        with tf.variable_scope('actor'):
            h1 = tf.layers.dense(self.state_in, N_HIDDEN_1, tf.nn.relu,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='H1')
            h2 = tf.layers.dense(h1, N_HIDDEN_2, tf.nn.relu,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='H2')
            h3 = tf.layers.dense(h2, self.dim_action, tf.nn.sigmoid,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='actor')
            self.a_predict = 2*self.action_limit*h3 - self.action_limit
            
        with tf.variable_scope('actor_t'):
            h1 = tf.layers.dense(self.state_in, N_HIDDEN_1, tf.nn.relu,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='H1_t')
            h2 = tf.layers.dense(h1, N_HIDDEN_2, tf.nn.relu,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='H2_t')
            h3 = tf.layers.dense(h2, self.dim_action, tf.nn.sigmoid,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='target')
            self.a_target = 2*self.action_limit*h3 - self.action_limit
            
        self.weights_a  = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor')
        self.weights_at = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor_t')
        
    def create_critic(self):
        tf.set_random_seed(self.seed)
        sXa = tf.concat([self.state_in, self.action_in], axis=1)
        with tf.variable_scope('critic'):
            h1 = tf.layers.dense(sXa, N_HIDDEN_1, tf.nn.relu,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='H1')
            h2 = tf.layers.dense(h1, N_HIDDEN_2, tf.nn.relu,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='H2')
            self.q_predict = tf.layers.dense(h2, 1,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='critic')
            
        with tf.variable_scope('critic_t'):
            h1 = tf.layers.dense(sXa, N_HIDDEN_1, tf.nn.relu,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='H1_t')
            h2 = tf.layers.dense(h1, N_HIDDEN_2, tf.nn.relu,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='H2_t')
            self.q_target = tf.layers.dense(h2, 1,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='target')
            
        self.weights_c  = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic')
        self.weights_ct = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic_t')
        
    def create_optimizer(self):
        self.c_error = self.q_predict-self.q_value_in
        self.c_cost = tf.reduce_mean(tf.pow(self.c_error,2))
        self.c_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_Q).minimize(self.c_cost)
        
        #q_pred = tf.reshape(self.q_predict, [-1])
        self.q_gradient = tf.gradients(self.q_predict, self.action_in)
        #self.q_gradient = tf.stop_gradient(self.q_gradient)
        
        self.a_gradients = tf.gradients(self.a_predict, self.weights_a, -self.q_gradient_in)
        self.a_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_A).apply_gradients(zip(self.a_gradients, self.weights_a))
    
    def create_update_operation(self):
        copy_net_ops = []
        for var, var_old in zip(self.weights_a, self.weights_at):
            copy_net_ops.append(var_old.assign(var))
        for var, var_old in zip(self.weights_c, self.weights_ct):
            copy_net_ops.append(var_old.assign(var))
        self.copy_net_ops = copy_net_ops
        
        update_anet_ops = []
        for var, var_old in zip(self.weights_a, self.weights_at):
            update_anet_ops.append(var_old.assign(TAU*var+(1-TAU)*var_old))
        self.update_anet_ops = update_anet_ops
        
        update_cnet_ops = []
        for var, var_old in zip(self.weights_c, self.weights_ct):
            update_cnet_ops.append(var_old.assign(TAU*var+(1-TAU)*var_old))
        self.update_cnet_ops = update_cnet_ops
        
    def init_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config,graph=self.g)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.copy_net_ops)
        
    def update_target_anet(self):
        self.sess.run(self.update_anet_ops)
        
    def update_target_qnet(self):
        self.sess.run(self.update_cnet_ops)
        
    def train_anetwork(self, state_t_batch, gradient_batch):
        return self.sess.run(self.a_optimizer,
                    #feed_dict={self.state_in:state_t_batch, self.action_in:action_t_batch})
                    #feed_dict={self.state_in:state_t_batch})
                    #feed_dict={self.q_gradient_in:gradient_batch})
                    feed_dict={self.state_in:state_t_batch, self.q_gradient_in:gradient_batch})
                    #feed_dict={self.state_in:state_t_batch and i: d for i, d in zip(self.q_gradient_in, gradient_batch)})
    
    def train_qnetwork(self, state_t_batch, action_t_batch, q_batch):
        return self.sess.run([self.c_error, self.c_cost, self.c_optimizer],
                     feed_dict={self.state_in:state_t_batch, self.action_in:action_t_batch, self.q_value_in:q_batch})
    
    def get_q_gradients(self, state_t_batch, action_t_batch):
        return self.sess.run(self.q_gradient, feed_dict={self.state_in:state_t_batch, self.action_in:action_t_batch})
    
    def get_a(self, state):
        a = self.sess.run(self.a_predict, feed_dict={self.state_in:np.reshape(state, [1,-1])})
        return a
    
    def get_ta(self, state_t):
        a = self.sess.run(self.a_target, feed_dict={self.state_in:state_t})
        return a
    
    def get_q(self, state_t, action_t):
        qs = self.sess.run(self.q_predict, feed_dict={self.state_in:state_t, self.action_in:action_t})
        return qs
    
    def get_tq(self, state_t, action_t):
        qs = self.sess.run(self.q_target, feed_dict={self.state_in:state_t, self.action_in:action_t})
        return qs
    
    def save_network(self, _name):
        self.saver.save(self.sess, "./weights/model_qweight_"+_name+".ckpt")
        print("* Successfully save the networks.")
        
    def load_network(self, _name):
        self.saver.restore(self.sess, "./weights/model_qweight_"+_name+".ckpt")
        print("* Successfully load the networks.")
