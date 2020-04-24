from sklearn.utils import shuffle
from collections import deque
from copy import deepcopy
import tensorflow as tf
from mpi4py import MPI
import numpy as np
import pickle
import random
import copy
import time
import os

EPS = 1e-10

class Agent:
    def __init__(self, env, args, id=0):
        self.id = 0
        self.env = env
        self.name = args['agent_name']
        self.checkpoint_dir='{}/checkpoint'.format(args['env_name'])
        self.discount_factor = args['discount_factor']
        self.state_dim = env.observation_space.shape[0]
        try:
            self.action_dim = env.action_space.shape[0]
            self.action_bound_min = env.action_space.low
            self.action_bound_max = env.action_space.high
        except:
            self.action_dim = 1
            self.action_bound_min = - 1.0
            self.action_bound_max = 1.0
        self.hidden1_units = args['hidden1']
        self.hidden2_units = args['hidden2']
        self.v_lr = args['v_lr']
        self.p_lr = args['p_lr']
        self.value_epochs = args['value_epochs']
        self.policy_epochs = args['policy_epochs']
        self.clip_value = args['clip_value']
        self.gae_coeff = args['gae_coeff']
        self.ent_coeff = args['ent_coeff']

        with tf.variable_scope(self.name):
            #placeholder
            self.states = tf.placeholder(tf.float32, [None, self.state_dim], name='State')
            self.actions = tf.placeholder(tf.float32, [None, self.action_dim], name='Action')
            self.targets = tf.placeholder(tf.float32, [None], name='targets')
            self.gaes = tf.placeholder(tf.float32, [None], name='gaes')
            self.old_std = tf.placeholder(tf.float32, [None, self.action_dim], name='old_std')
            self.old_mean = tf.placeholder(tf.float32, [None, self.action_dim], name='old_mean')
            self.log_prob_old = tf.placeholder(tf.float32, [None,], name='log_prob_old')

            #policy & value & kl & entropy
            self.mean, self.std = self.build_policy_model('policy')
            self.value = self.build_value_model('value')
            self.kl, self.entropy = self.get_kl_and_entropy()

            #action
            self.norm_noise_action = self.mean + tf.multiply(tf.random_normal(tf.shape(self.mean)), self.std)
            self.sample_noise_action = self.unnormalize_action(self.norm_noise_action)
            self.norm_action = self.mean
            self.sample_action = self.unnormalize_action(self.norm_action)

            #value loss
            v_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name+'/value')
            self.v_loss = 0.5*tf.square(self.targets - self.value)
            self.v_loss = tf.reduce_mean(self.v_loss)
            v_optimizer = tf.train.AdamOptimizer(learning_rate=self.v_lr)
            v_gradients = tf.gradients(self.v_loss, v_vars)
            self.v_train_op = v_optimizer.apply_gradients(zip(v_gradients, v_vars))

            #policy optimizer
            norm_actions = self.normalize_action(self.actions)
            p_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name+'/policy')
            self.log_prob = - tf.reduce_sum(tf.log(self.std + EPS) + 0.5*np.log(2*np.pi) + tf.squared_difference(norm_actions, self.mean) / (2 * tf.square(self.std) + EPS), axis=1)
            ratios = tf.exp(self.log_prob - self.log_prob_old)
            clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1 - self.clip_value, clip_value_max=1 + self.clip_value)
            loss_clip = tf.minimum(tf.multiply(self.gaes, ratios), tf.multiply(self.gaes, clipped_ratios))
            self.p_loss = -(tf.reduce_mean(loss_clip) + self.ent_coeff*self.entropy)
            p_optimizer = tf.train.AdamOptimizer(learning_rate=self.p_lr)
            p_gradients = tf.gradients(self.p_loss, p_vars)
            self.p_train_op = p_optimizer.apply_gradients(zip(p_gradients, p_vars))

            #define sync operator
            self.train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
            self.sync_vars_ph = []
            self.sync_op = []
            for v in self.train_vars:
                self.sync_vars_ph.append(tf.placeholder(tf.float32, shape=v.get_shape()))
                self.sync_op.append(v.assign(self.sync_vars_ph[-1]))

            if self.id == 0:
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
            else:
                config = tf.ConfigProto(device_count={'GPU': 0})
            self.sess = tf.Session(config=config)
            self.load()


    #define sync operator
    def sync(self):
        if self.id == 0:
            train_vars = self.sess.run(self.train_vars)
        else :
            train_vars = None
        train_vars = MPI.COMM_WORLD.bcast(train_vars, root=0)
        if self.id == 0:
            return
        feed_dict = dict({s_ph: s for s_ph, s in zip(self.sync_vars_ph, train_vars)})
        self.sess.run(self.sync_op, feed_dict=feed_dict)

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

    def build_policy_model(self, name='policy'):
        with tf.variable_scope(name):
            model = tf.layers.dense(self.states, self.hidden1_units, activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            model = tf.layers.batch_normalization(model)
            model = tf.nn.tanh(model)
            model = tf.layers.dense(model, self.hidden2_units, activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            model = tf.layers.batch_normalization(model)
            model = tf.nn.tanh(model)
            mean = tf.layers.dense(model, self.action_dim, activation=tf.tanh, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            logits_std = tf.get_variable("logits_std",shape=(self.action_dim),initializer=tf.random_normal_initializer(mean=-1.0,stddev=0.02)) # 0.1정도로 initialize
            std = tf.ones_like(mean)*tf.nn.softplus(logits_std)
        return mean, std

    def build_value_model(self, name='value'):
        with tf.variable_scope(name):
            inputs = self.states
            model = tf.layers.dense(inputs, self.hidden1_units, activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            model = tf.layers.batch_normalization(model)
            model = tf.nn.tanh(model)
            model = tf.layers.dense(model, self.hidden2_units, activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            model = tf.layers.batch_normalization(model)
            model = tf.nn.tanh(model)
            model = tf.layers.dense(model, 1, activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            model = tf.reshape(model, [-1])
            return model

    def get_kl_and_entropy(self):
        mean, std = self.mean, self.std
        old_mean, old_std = self.old_mean, self.old_std
        log_std_old = tf.log(old_std + EPS)
        log_std_new = tf.log(std + EPS)
        frac_std_old_new = old_std/(std + EPS)
        kl = tf.reduce_mean(log_std_new - log_std_old + 0.5*tf.square(frac_std_old_new) + 0.5*tf.square((mean - old_mean)/(std + EPS))- 0.5)
        entropy = tf.reduce_mean(log_std_new + 0.5 + 0.5*np.log(2*np.pi))
        return kl, entropy

    def get_action(self, state, is_train):
        if is_train:
            [[action], [value]] = self.sess.run([self.sample_noise_action, self.value], feed_dict={self.states:[state]})
        else:
            [[action], [value]] = self.sess.run([self.sample_action, self.value], feed_dict={self.states:[state]})
        clipped_action = np.clip(action, self.action_bound_min, self.action_bound_max)
        return action, clipped_action, value

    def train(self, trajs):
        states = trajs[0]
        actions = trajs[1]
        targets = trajs[2]
        next_states = trajs[3]
        rewards = trajs[4]
        gaes = trajs[5]
        old_means, old_stds, old_log_probs = self.sess.run([self.mean, self.std, self.log_prob], 
                                                    feed_dict={self.states:states, self.actions:actions})

        #POLICY update
        p_s, p_a, p_old_m, p_old_s, p_old_l_p, p_g = shuffle(states, actions, old_means, old_stds, old_log_probs, gaes, random_state=0)
        for _ in range(self.policy_epochs):
            p_s, p_a, p_old_m, p_old_s, p_old_l_p, p_g = shuffle(p_s, p_a, p_old_m, p_old_s, p_old_l_p, p_g, random_state=0)
            self.sess.run(self.p_train_op, feed_dict={
                    self.states:p_s,
                    self.actions:p_a,
                    self.old_mean:p_old_m,
                    self.old_std:p_old_s,
                    self.log_prob_old:p_old_l_p,
                    self.gaes:p_g})
        p_loss, kl, entropy = self.sess.run([self.p_loss, self.kl, self.entropy], feed_dict={
            self.states:states, 
            self.actions:actions,
            self.old_mean:old_means, 
            self.old_std:old_stds,
            self.log_prob_old:old_log_probs,
            self.gaes:gaes})

        #VALUE update
        v_s, v_t = shuffle(states, targets, random_state=0)
        for _ in range(self.value_epochs):
            v_s, v_t = shuffle(v_s, v_t, random_state=0)
            self.sess.run(self.v_train_op, feed_dict={
                    self.states:v_s,
                    self.targets:v_t})
        v_loss = self.sess.run(self.v_loss, feed_dict={self.states:states, self.targets:targets})

        return p_loss, v_loss, kl, entropy

    def get_gaes_targets(self, rewards, values, next_values):
        deltas = np.array(rewards) + self.discount_factor*np.array(next_values) - np.array(values)
        gaes = deepcopy(deltas)
        targets = np.zeros_like(rewards)
        ret = 0
        for t in reversed(range(len(gaes))):
            if t < len(gaes) - 1:
                gaes[t] = gaes[t] + self.discount_factor*self.gae_coeff*gaes[t + 1]
            ret = rewards[t] + self.discount_factor*ret
            targets[t] = ret
        return gaes, targets

    def save(self):
        self.saver.save(self.sess, self.checkpoint_dir+'/model.ckpt')
        print('[{}] save success!'.format(self.name))

    def load(self):
        self.saver = tf.train.Saver(var_list= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name))

        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print('[{}] success to load model!'.format(self.name))
        else:
            self.sess.run(tf.global_variables_initializer())
            print('[{}] fail to load model...'.format(self.name))
        