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
        env_name = env_name.split('-')[0]
        self.checkpoint_dir='{}/checkpoint'.format(env_name)
        self.discount_factor = gamma
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.observation_space.shape[0]
        self.hidden1_units = 128
        self.hidden2_units = 128
        self.v_lr = 1e-3
        self.p_lr = 1e-4

        with tf.variable_scope(self.name):
            self.states = tf.placeholder(tf.float32, [None, self.state_dim], name='State')
            self.actions = tf.placeholder(tf.float32, [None, self.action_dim], name='Action')
            self.targets = tf.placeholder(tf.float32, [None], name='targets')

            self.mean, self.std = self.build_policy_model('policy')
            self.value = self.build_value_model('value')

            config = tf.ConfigProto(device_count={'GPU': 0})
            self.sess = tf.Session(config=config)
            #self.load()

    def build_policy_model(self, name='policy'):
        with tf.variable_scope(name):
            model = tf.layers.dense(self.states, self.hidden1_units, activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            model = tf.layers.batch_normalization(model)
            model = tf.nn.relu(model)
            model = tf.layers.dense(model, self.hidden2_units, activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            model = tf.layers.batch_normalization(model)
            model = tf.nn.relu(model)

            mean = tf.layers.dense(model, self.action_dim, activation=tf.tanh, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            logits_std = tf.get_variable("logits_std",shape=(self.action_dim),initializer=tf.random_normal_initializer(mean=-1.0,stddev=0.02)) # 0.1정도로 initialize
            std = tf.ones_like(mean)*tf.nn.softplus(logits_std)
        return mean, std

    def build_value_model(self, name='value'):
        with tf.variable_scope(name):
            model = tf.layers.dense(self.states, self.hidden1_units, activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            model = tf.layers.batch_normalization(model)
            model = tf.nn.relu(model)
            model = tf.layers.dense(model, self.hidden2_units, activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            model = tf.layers.batch_normalization(model)
            model = tf.nn.relu(model)
            model = tf.layers.dense(model, 1, activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            model = tf.reshape(model, [-1])
            return model

    def save(self):
        self.saver.save(self.sess, self.checkpoint_dir+'/model.ckpt')
        print('save 성공!')

    def load(self):
        self.saver = tf.train.Saver(var_list= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print('success to load model!')
        else:
            self.sess.run(tf.global_variables_initializer())
            print('fail to load model...')
