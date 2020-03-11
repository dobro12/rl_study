from collections import deque
import tensorflow as tf
import numpy as np
import itertools
import pickle
import random
import copy
import time
import os

class OUActionNoise: # OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

class Agent:
    def __init__(self, env, args):
        self.env = env
        self.name = args['agent_name']
        self.checkpoint_dir='{}/checkpoint'.format(args['env_name'])
        self.discount_factor = args['discount_factor']
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound_min = env.action_space.low
        self.action_bound_max = env.action_space.high
        self.hidden1_units = args['hidden1']
        self.hidden2_units = args['hidden2']
        self.v_lr = args['v_lr']
        self.p_lr = args['p_lr']
        self.soft_update = args['soft_update']
        self.replay_memory = deque(maxlen=int(1e5))

        self.train_start = int(1e3)
        self.batch_size = int(1e3)
        self.epsilon = 1.0
        self.epsilon_decay = 0.9999
        self.min_epsilon = 0.01
        self.se_rate = 0.3
        self.actor_noise = OUActionNoise(mu=np.zeros(self.action_dim), sigma=self.se_rate*self.epsilon)

        with tf.variable_scope(self.name):
            #placeholder
            self.states = tf.placeholder(tf.float32, [None, self.state_dim], name='State')
            self.actions = tf.placeholder(tf.float32, [None, self.action_dim], name='Action')
            self.targets = tf.placeholder(tf.float32, [None,], name='targets')

            #action & value
            self.policy = self.build_policy_model('policy')
            self.value = self.build_value_model('value')
            self.target_policy = self.build_policy_model('target_policy')
            self.target_value = self.build_value_model('target_value')
            self.sample_action = self.unnormalize_action(self.policy)
            self.norm_action = self.normalize_action(self.actions)

            #policy loss
            self.q_action_gradient = tf.gradients(self.value, self.actions)
            self.p_loss = -tf.multiply(self.policy, self.q_action_gradient)
            p_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name+'/policy')
            p_optimizer = tf.train.AdamOptimizer(learning_rate=self.p_lr)
            self.p_train_op = p_optimizer.minimize(self.p_loss, var_list=p_vars)

            #value loss
            self.v_loss = 0.5*tf.square(self.targets - self.value)
            self.v_loss = tf.reduce_sum(self.v_loss)
            v_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name+'/value')
            v_optimizer = tf.train.AdamOptimizer(learning_rate=self.v_lr)
            self.v_train_op = v_optimizer.minimize(self.v_loss, var_list=v_vars)

            #assign operator
            self.soft_update_op = self.build_soft_update_op()
            self.init_target_op = self.build_init_target_op()

            #make session and load model
            self.sess = tf.Session()
            self.load()

    def build_policy_model(self, name='policy'):
        with tf.variable_scope(name):
            model = tf.layers.dense(self.states, self.hidden1_units, activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            model = tf.layers.batch_normalization(model)
            model = tf.nn.tanh(model)
            model = tf.layers.dense(model, self.hidden2_units, activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            model = tf.layers.batch_normalization(model)
            model = tf.nn.tanh(model)
            model = tf.layers.dense(model, self.action_dim, activation=tf.tanh, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
        return model

    def build_value_model(self, name='value'):
        with tf.variable_scope(name):
            inputs = tf.concat([self.states, self.actions], axis=1)
            model = tf.layers.dense(inputs, self.hidden1_units, activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            model = tf.layers.batch_normalization(model)
            model = tf.nn.tanh(model)
            model = tf.layers.dense(model, self.hidden2_units, activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            model = tf.layers.batch_normalization(model)
            model = tf.nn.tanh(model)
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

    def build_soft_update_op(self):
        copy_op = []

        main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name+'/policy')
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name+'/target_policy')
        for main_var, target_var in zip(main_vars, target_vars):
            copy_op.append(target_var.assign( tf.multiply( main_var.value(), self.soft_update) + tf.multiply( target_var.value(), 1.0 - self.soft_update) ))

        main_vars2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name+'/value')
        target_vars2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name+'/target_value')
        for main_var, target_var in zip(main_vars2, target_vars2):
            copy_op.append(target_var.assign( tf.multiply( main_var.value(), self.soft_update) + tf.multiply( target_var.value(), 1.0 - self.soft_update) ))
        return copy_op

    def build_init_target_op(self):
        copy_op = []

        main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name+'/policy')
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name+'/target_policy')
        for main_var, target_var in zip(main_vars, target_vars):
            copy_op.append(target_var.assign(main_var.value()))

        main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name+'/value')
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name+'/target_value')
        for main_var, target_var in zip(main_vars, target_vars):
            copy_op.append(target_var.assign(main_var.value()))
        return copy_op

    def get_action(self, state, is_train):
        [action] = self.sess.run(self.sample_action, feed_dict={self.states:[state]})
        if is_train:
            action += np.random.normal(size=action.shape, scale=self.epsilon)
            #action += self.actor_noise()
            #self.action_noise 이용해서도 짜보자
        action = np.clip(action, self.action_bound_min, self.action_bound_max)
        return action

    def get_value(self, state, action):
        [value] = self.sess.run(self.value, feed_dict={self.states:[state], self.actions:[action]})
        return value

    def train(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
            self.actor_noise.sigma = self.se_rate*self.epsilon

        mini_batch = random.sample(self.replay_memory, self.batch_size)
        states = [batch[0] for batch in mini_batch]
        actions = [batch[1] for batch in mini_batch]
        rewards = [batch[2] for batch in mini_batch]
        next_states = [batch[4] for batch in mini_batch]

        target_actions = self.sess.run(self.target_policy, feed_dict={self.states:next_states})
        next_values = self.sess.run(self.target_value, feed_dict={self.states:next_states, self.actions:target_actions})
        actions = self.sess.run(self.norm_action, feed_dict={self.actions:actions})
        mu_actions = self.sess.run(self.policy, feed_dict={self.states:states})

        '''
        targets = []
        for i in range(self.batch_size):
            reward = mini_batch[i][2]
            done = mini_batch[i][3]
            if done:
                targets.append(reward)
            else:
                targets.append(reward + self.discount_factor*next_values[i])
        '''
        targets = np.array(rewards) + self.discount_factor*next_values
        #a = itertools.islice(self.replay_memory, len(self.replay_memory)-100, len(self.replay_memory))
        #r = [batch[2] for batch in a]
        #print(np.mean(targets),np.mean(next_values), np.mean(rewards), np.mean(r))

        _, v_loss = self.sess.run([self.v_train_op, self.v_loss], feed_dict={self.states:states, self.actions:actions, self.targets:targets})
        _, p_loss = self.sess.run([self.p_train_op, self.p_loss], feed_dict={self.states:states, self.actions:mu_actions})
        self.sess.run(self.soft_update_op)

        return v_loss, p_loss

    def save(self):
        self.saver.save(self.sess, self.checkpoint_dir+'/model.ckpt')
        with open('{}/replay.pkl'.format(self.checkpoint_dir), 'wb') as f:
            pickle.dump([self.epsilon, self.replay_memory], f)
        print('save 성공!')

    def load(self):
        self.saver = tf.train.Saver(var_list= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            with open('{}/replay.pkl'.format(self.checkpoint_dir), 'rb') as f:
                self.epsilon, self.replay_memory = pickle.load(f)
                self.actor_noise.sigma = self.se_rate*self.epsilon
            print('success to load model!')
        else:
            self.sess.run(tf.global_variables_initializer())
            #initialize target network
            self.sess.run(self.init_target_op)
            print('fail to load model...')
        
