from collections import deque
import tensorflow as tf
import numpy as np
import random
import copy
import time
import os

class Agent:
    def __init__(self, agent_name, env, env_name, gamma):
        self.env = env
        self.name = agent_name
        self.checkpoint_dir='{}/checkpoint'.format(env_name)
        self.discount_factor = gamma
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        #self.action_dim = 1
        self.action_bound_min = env.action_space.low
        self.action_bound_max = env.action_space.high
        #self.action_bound_min = np.array([-1.0])
        #self.action_bound_max = np.array([1.0])
        self.hidden1_units = 128
        self.hidden2_units = 256
        self.v_lr = 1e-3
        self.p_lr = 1e-4 #1e-4
        self.init_std = 0.0 #1.0

        with tf.variable_scope(self.name):
            #placeholder
            self.states = tf.placeholder(tf.float32, [None, self.state_dim], name='State')
            self.actions = tf.placeholder(tf.float32, [None, self.action_dim], name='Action')
            self.targets = tf.placeholder(tf.float32, [None], name='targets')

            #action & value
            self.mean, self.std = self.build_policy_model('policy')
            self.value = self.build_value_model('value')
            self.norm_noise_action = self.mean + tf.random_normal(tf.shape(self.mean))*self.std
            self.sample_noise_action = self.unnormalize_action(self.norm_noise_action)
            self.norm_action = self.mean
            self.sample_action = self.unnormalize_action(self.norm_action)
            self.entropy = tf.reduce_mean(self.std + 0.5 + 0.5*np.log(2*np.pi))

            #policy loss
            norm_actions = self.normalize_action(self.actions)
            self.p_loss = tf.reduce_sum(tf.log(self.std + 1e-10) + tf.squared_difference(norm_actions, self.mean)/(2*tf.square(self.std) + 1e-10), axis=1)
            #self.p_loss = tf.reduce_mean(self.p_loss*(self.targets - self.value))
            self.p_loss = tf.reduce_sum(self.p_loss*(self.targets - self.value))
            p_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name+'/policy')
            p_optimizer = tf.train.AdamOptimizer(learning_rate=self.p_lr)
            self.p_gradients = tf.gradients(self.p_loss, p_vars)
            self.p_train_op = p_optimizer.apply_gradients(zip(self.p_gradients, p_vars))

            #value loss
            self.v_loss = 0.5*tf.square(self.targets - self.value)
            #self.v_loss = tf.reduce_mean(self.v_loss)
            self.v_loss = tf.reduce_sum(self.v_loss)
            v_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name+'/value')
            v_optimizer = tf.train.AdamOptimizer(learning_rate=self.v_lr)
            self.v_gradients = tf.gradients(self.v_loss, v_vars)
            self.v_train_op = v_optimizer.apply_gradients(zip(self.v_gradients, v_vars))

            #assign operator
            self.train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
            self.target_vars = []
            self.copy_op = []
            for train_var in self.train_vars:
                target_var = tf.Variable(train_var)
                self.target_vars.append(target_var)
                self.copy_op.append(train_var.assign(target_var.value()))

            config = tf.ConfigProto(device_count={'GPU': 0})
            self.sess = tf.Session(config=config)
            if self.name == 'global':
                self.load()
            else:
                self.sess.run(tf.global_variables_initializer())

    def build_policy_model(self, name='policy'):
        with tf.variable_scope(name):
            model = tf.layers.dense(self.states, self.hidden1_units, activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            model = tf.layers.batch_normalization(model)
            model = tf.nn.tanh(model)
            model = tf.layers.dense(model, self.hidden2_units, activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            model = tf.layers.batch_normalization(model)
            model = tf.nn.tanh(model)

            mean = tf.layers.dense(model, self.action_dim, activation=tf.tanh, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            logits_std = tf.get_variable("logits_std",shape=(self.action_dim),initializer=tf.random_normal_initializer(mean=self.init_std, stddev=0.02)) # 0.1정도로 initialize
            std = tf.ones_like(mean)*tf.nn.softplus(logits_std)
            #std = 0.1
        return mean, std

    def build_value_model(self, name='value'):
        with tf.variable_scope(name):
            model = tf.layers.dense(self.states, self.hidden1_units, activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            model = tf.layers.batch_normalization(model)
            model = tf.nn.tanh(model)
            model = tf.layers.dense(model, self.hidden2_units, activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            model = tf.layers.batch_normalization(model)
            model = tf.nn.tanh(model)
            model = tf.layers.dense(model, self.hidden2_units, activation=tf.nn.tanh, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            model = tf.layers.dense(model, 1, activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            model = tf.reshape(model, [-1])
            return model

    def normalize_action(self, a):
        temp_a = 2.0/(self.action_bound_max - self.action_bound_min)
        temp_b = (self.action_bound_max + self.action_bound_min)/(self.action_bound_min - self.action_bound_max)
        temp_a = tf.ones_like(a)*temp_a
        temp_b = tf.ones_like(a)*temp_b
        return temp_a*a + temp_b

    def unnormalize_action(self, a):
        temp_a = (self.action_bound_max - self.action_bound_min)/2.0
        temp_b = (self.action_bound_max + self.action_bound_min)/2.0
        temp_a = tf.ones_like(a)*temp_a
        temp_b = tf.ones_like(a)*temp_b
        return temp_a*a + temp_b

    def calc_gradient(self, states, actions, targets):
        p_grad, p_loss, v_grad, v_loss, entropy = self.sess.run([self.p_gradients, self.p_loss, self.v_gradients, self.v_loss, self.entropy], feed_dict={
            self.states:states,
            self.actions:actions,
            self.targets:targets
        })
        return p_grad, p_loss, v_grad, v_loss, entropy

    def update_with_gradients(self, p_grad, v_grad):
        self.sess.run(self.p_train_op, feed_dict={
            i: d for i, d in zip(self.p_gradients, p_grad)
        })
        self.sess.run(self.v_train_op, feed_dict={
            i: d for i, d in zip(self.v_gradients, v_grad)
        })

    def get_parameter_value(self):
        return self.sess.run(self.train_vars)
    
    def update_parameter(self, target_agent):
        target_vars = target_agent.get_parameter_value()
        self.sess.run(self.copy_op, feed_dict={
            i:d for i,d in zip(self.target_vars, target_vars)
        })

    def get_action(self, state, is_train):
        if is_train:
            [action] = self.sess.run(self.sample_noise_action, feed_dict={self.states:[state]})
        else:
            [action] = self.sess.run(self.sample_action, feed_dict={self.states:[state]})
        action = np.clip(action, self.action_bound_min, self.action_bound_max)
        return action

    def get_value(self, state):
        [value] = self.sess.run(self.value, feed_dict={self.states:[state]})
        return value

    def save(self):
        self.saver.save(self.sess, self.checkpoint_dir+'/model.ckpt')
        print('save 성공!')

    def load(self):
        self.saver = tf.train.Saver(var_list= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name))

        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print('success to load model!')
        else:
            self.sess.run(tf.global_variables_initializer())
            print('fail to load model...')
