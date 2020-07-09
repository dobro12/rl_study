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
import gym
import os

EPS = 1e-10

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class Agent:
    def __init__(self, env, args, id=0):
        self.id = id
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
        self.q_lr = args['q_lr']
        self.p_lr = args['p_lr']
        self.alpha = args.get('alpha', 0.2)
        self.soft_update = args.get('soft_update', 0.005)
        self.batch_size = args.get('batch_size', 100)
        self.replay_memory = deque(maxlen=int(args.get('replay_memory', 1e5)))
        self.is_loaded = False

        with tf.variable_scope(self.name):
            #placeholder
            self.states = tf.placeholder(tf.float32, [None, self.state_dim], name='states')
            self.next_states = tf.placeholder(tf.float32, [None, self.state_dim], name='next_states')
            self.actions = tf.placeholder(tf.float32, [None, self.action_dim], name='actions')
            self.targets = tf.placeholder(tf.float32, [None], name='targets')
            self.dones = tf.placeholder(tf.float32, [None], name='dones')
            self.rewards = tf.placeholder(tf.float32, [None], name='rewards')

            #policy & q_value
            self.mean, self.log_std, self.std = self.build_policy_model(self.states, 'policy')
            self.next_mean, self.next_log_std, self.next_std = self.build_policy_model(self.next_states, 'policy', reuse=True)
            self.q1 = self.build_q_value_model(self.states, self.actions, 'q1')
            self.q2 = self.build_q_value_model(self.states, self.actions, 'q2')

            #action
            gaussian_noise = tf.random_normal(tf.shape(self.mean))
            self.norm_noise_action = self.mean + tf.multiply(gaussian_noise, self.std)
            self.norm_action = self.mean
            self.log_prob = -tf.reduce_sum(self.log_std + 0.5*np.log(2*np.pi) + 0.5*tf.square(gaussian_noise), axis=1)

            #apply squashing
            self.log_prob -= tf.reduce_sum(2*(np.log(2.0) - self.norm_noise_action - tf.nn.softplus(-2*self.norm_noise_action)) + tf.log(self.action_bound_max), axis=1)
            self.norm_noise_action = tf.nn.tanh(self.norm_noise_action)
            self.norm_action = tf.nn.tanh(self.norm_action)
            self.sample_noise_action = self.unnormalize_action(self.norm_noise_action)
            self.sample_action = self.unnormalize_action(self.norm_action)

            #next action
            next_mean, next_log_std, next_std = self.build_policy_model(self.next_states, 'policy', reuse=True)
            next_gaussian_noise = tf.random_normal(tf.shape(next_mean))
            next_norm_noise_action = next_mean + tf.multiply(next_gaussian_noise, next_std)
            self.log_prob_next = -tf.reduce_sum(next_log_std + 0.5*np.log(2*np.pi) + 0.5*tf.square(next_gaussian_noise), axis=1)
            self.log_prob_next -= tf.reduce_sum(2*(np.log(2.0) - next_norm_noise_action - tf.nn.softplus(-2*next_norm_noise_action)) + tf.log(self.action_bound_max), axis=1)
            next_norm_noise_action = tf.nn.tanh(next_norm_noise_action)
            next_sample_noise_action = self.unnormalize_action(next_norm_noise_action)

            # target Q value
            self.q1_target = self.build_q_value_model(self.next_states, next_sample_noise_action, 'target_q1')
            self.q2_target = self.build_q_value_model(self.next_states, next_sample_noise_action, 'target_q2')

            #q_pi
            self.q1_pi = self.build_q_value_model(self.states, self.sample_noise_action, 'q1', reuse=True)
            self.q2_pi = self.build_q_value_model(self.states, self.sample_noise_action, 'q2', reuse=True)

            #policy loss
            p_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name+'/policy')
            self.p_loss = -tf.reduce_mean(tf.minimum(self.q1_pi, self.q2_pi) - self.alpha*self.log_prob)
            p_optimizer = tf.train.AdamOptimizer(learning_rate=self.p_lr)
            self.p_train_op = p_optimizer.minimize(self.p_loss, var_list=p_vars)

            #value loss
            q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name+'/q')
            self.target_value = self.rewards + self.discount_factor*(1 - self.dones)*(tf.minimum(self.q1_target, self.q2_target) - self.alpha*self.log_prob_next)
            self.target_value = tf.stop_gradient(self.target_value)
            self.q_loss = tf.reduce_mean(0.5*tf.square(self.target_value - self.q1))
            self.q_loss += tf.reduce_mean(0.5*tf.square(self.target_value - self.q2))
            q_optimizer = tf.train.AdamOptimizer(learning_rate=self.q_lr)
            self.q_train_op = q_optimizer.minimize(self.q_loss, var_list=q_vars)

            #target update
            name_pair_list = [['q1', 'target_q1'], ['q2', 'target_q2']]
            self.target_update_op = self.build_soft_update_op(name_pair_list)
            self.target_init_op = self.build_init_target_op(name_pair_list)

            #entropy
            self.entropy = self.get_entropy()

            #define sync operator
            self.train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
            self.sync_vars_ph = []
            self.sync_op = []
            for v in self.train_vars:
                self.sync_vars_ph.append(tf.placeholder(tf.float32, shape=v.get_shape()))
                self.sync_op.append(v.assign(self.sync_vars_ph[-1]))

            #session & load
            if self.id == 0:
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
            else:
                config = tf.ConfigProto(device_count={'GPU': 0})
            self.sess = tf.Session(config=config)
            self.load()

            #initialize target Q network
            if not self.is_loaded:
                self.sess.run(self.target_init_op)
                print("[{}] initialize target Q networks.".format(self.name))


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

    def build_policy_model(self, states, name='policy', reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            model = tf.layers.dense(states, self.hidden1_units, activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            model = tf.layers.batch_normalization(model)
            model = tf.nn.tanh(model)
            model = tf.layers.dense(model, self.hidden2_units, activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            model = tf.layers.batch_normalization(model)
            model = tf.nn.tanh(model)
            mean = tf.layers.dense(model, self.action_dim, activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            log_std = tf.layers.dense(model, self.action_dim, activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
            std = tf.exp(log_std)
        return mean, log_std, std

    def build_q_value_model(self, states, actions, name='value', reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            inputs = tf.concat([states, actions], axis=1)
            model = tf.layers.dense(inputs, self.hidden1_units, activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            model = tf.layers.batch_normalization(model)
            model = tf.nn.tanh(model)
            model = tf.layers.dense(model, self.hidden2_units, activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            model = tf.layers.batch_normalization(model)
            model = tf.nn.tanh(model)
            model = tf.layers.dense(model, 1, activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            model = tf.reshape(model, [-1])
            return model

    def get_entropy(self):
        #entropy = tf.reduce_mean(tf.reduce_sum(self.log_std + 0.5 + 0.5*np.log(2*np.pi), axis=1))
        entropy = tf.reduce_mean(self.log_std + 0.5 + 0.5*np.log(2*np.pi)) #원래 위에 식이 맞는데 이 식이 좀 더 직관적임
        return entropy

    def get_action(self, state, is_train):
        [[nois_action], [action]] = self.sess.run([self.sample_noise_action, self.sample_action], feed_dict={self.states:[state]})
        if is_train:
            return nois_action
        else:
            return action

    def train(self):
        sampled_trajectory = random.sample(self.replay_memory, self.batch_size)
        states = [t[0] for t in sampled_trajectory]
        actions = [t[1] for t in sampled_trajectory]
        rewards = [t[2] for t in sampled_trajectory]
        dones = [t[3] for t in sampled_trajectory]
        next_states = [t[4] for t in sampled_trajectory]

        #VALUE update
        feed_dict = {self.states:states, self.actions:actions, self.rewards:rewards, self.dones:dones, self.next_states:next_states}
        _, q_loss = self.sess.run([self.q_train_op, self.q_loss], feed_dict=feed_dict)
        #POLICY update
        _, p_loss, entropy = self.sess.run([self.p_train_op, self.p_loss, self.entropy], feed_dict={self.states:states})
        #TARGET update
        self.sess.run(self.target_update_op)

        return p_loss, q_loss, entropy

    def build_soft_update_op(self, name_pair_list):
        copy_op = []

        for name_pair in name_pair_list:
            orig_name, target_name = name_pair
            main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name + '/' + orig_name)
            target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name + '/' + target_name)
            for main_var, target_var in zip(main_vars, target_vars):
                copy_op.append(target_var.assign( tf.multiply( main_var.value(), self.soft_update) + tf.multiply( target_var.value(), 1.0 - self.soft_update) ))
        return copy_op

    def build_init_target_op(self, name_pair_list):
        copy_op = []

        for name_pair in name_pair_list:
            orig_name, target_name = name_pair
            main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name + '/' + orig_name)
            target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name + '/' + target_name)
            for main_var, target_var in zip(main_vars, target_vars):
                copy_op.append(target_var.assign(main_var.value()))
        return copy_op

    def save(self):
        self.saver.save(self.sess, self.checkpoint_dir+'/model.ckpt')
        with open('{}/replay.pkl'.format(self.checkpoint_dir), 'wb') as f:
            pickle.dump(self.replay_memory, f)
        print('[{}] save success!'.format(self.name))

    def load(self):
        self.saver = tf.train.Saver(var_list= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name))

        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            with open('{}/replay.pkl'.format(self.checkpoint_dir), 'rb') as f:
                self.replay_memory = pickle.load(f)
            print('[{}] success to load model!'.format(self.name))
            self.is_loaded = True
        else:
            self.sess.run(tf.global_variables_initializer())
            print('[{}] fail to load model...'.format(self.name))
            self.is_loaded = False
        

if __name__ == "__main__":
    env_name = 'HalfCheetah-v2'

    save_name = env_name.split('-')[0]
    agent_args = {'agent_name':'SAC',
                'env_name':save_name,
                'discount_factor':0.99,
                'hidden1':64,
                'hidden2':64,
                'q_lr':1e-3,
                'p_lr':1e-4,
                'value_epochs':80,
                'policy_epochs':10,
                'clip_value':0.2,
                'gae_coeff':0.97,
                'ent_coeff':0.0,
                }

    env = gym.make(env_name)
    agent = Agent(env, agent_args)
