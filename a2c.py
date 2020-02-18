from collections import deque
import numpy as np
import tensorflow as tf
import random
import copy
import time
import os

class Agent:
    def __init__(self, env):
        self.name = 'A2C'
        self.critic_lr = 0.005
        self.actor_lr = 0.001
        self.discount_factor = 0.99
        self.n_action = env.action_space.n
        self.state_dim = env.observation_space.shape[0]
        self.env = env
        self.save_freq = 100

        self.X = tf.placeholder(tf.float32, [None, self.state_dim], name='X')
        self.A = tf.placeholder(tf.int32, [None], name='Action')
        self.advantage = tf.placeholder(tf.float32, [None, self.n_action], name='Advantage')
        self.Y = tf.placeholder(tf.float32, [None], name='Y')

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.critic, self.train_critic_op = self._build_critic()
        self.actor, self.train_actor_op = self._build_actor()

        self.sess = tf.Session()
        self.load()

        print(self.sess.run(self.global_step))

    def _build_actor(self, name='actor'):
        with tf.variable_scope(name):
            model = tf.layers.dense(self.X, 24, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            #model = tf.layers.dense(model, 24, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            model = tf.layers.dense(model, self.n_action, activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            policy = tf.nn.softmax(model)

        cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=self.advantage))

        model_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        with tf.variable_scope(name):
            train_op = tf.train.AdamOptimizer(self.actor_lr, beta1=0.5, beta2=0.999).minimize(cost, var_list=model_vars)

        return policy, train_op

    def _build_critic(self, name='critic'):
        with tf.variable_scope(name):
            model = tf.layers.dense(self.X, 24, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            model = tf.layers.dense(model, 24, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            Q = tf.layers.dense(model, 1, activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))

        one_hot = tf.one_hot(self.A, self.n_action, 1.0, 0.0)
        Q_value = tf.reduce_sum(tf.multiply(one_hot, Q), axis=1)
        cost = tf.reduce_mean(tf.square(self.Y - Q_value))

        model_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        with tf.variable_scope(name):
            train_op = tf.train.AdamOptimizer(self.critic_lr, beta1=0.5, beta2=0.999).minimize(cost, var_list=model_vars, global_step=self.global_step)

        return Q, train_op

    def get_action(self, state):
        [policy] = self.sess.run(self.actor, feed_dict={self.X:[state]})
        return np.random.choice(self.n_action, 1, p=policy)[0]

    def _train(self, state, action, reward, done, next_state):
        [[Value]] = self.sess.run(self.critic, feed_dict={self.X:[state]})
        [[next_Value]] = self.sess.run(self.critic, feed_dict={self.X:[next_state]})
        
        critic_target = None
        actor_target = np.zeros(self.n_action)
        if done:
            critic_target = reward
            actor_target[action] = reward - Value
        else:
            critic_target = reward + self.discount_factor*next_Value
            actor_target[action] = reward + self.discount_factor*next_Value - Value

        self.sess.run(self.train_actor_op, feed_dict={self.X:[state], self.advantage:[actor_target]})
        self.sess.run(self.train_critic_op, feed_dict={self.X:[state], self.Y:[critic_target], self.A:[action]})

    def train(self, episodes=10000):
        count = 0
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            accumulate = 0

            while not done:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                reward = reward if not done or accumulate == 499 else -100
                self._train(state, action, reward, done, next_state)

                state = next_state
                accumulate += reward

            if accumulate == 500:
                count+=1
            else:
                count = 0
            if count >= 5:
                self.save()
                print('학습종료')
                return

            print(episode, accumulate+100)
            if (episode+1)%self.save_freq == 0:
                self.save()
        
    def test(self):
        self.epsilon = 0.0
        state = self.env.reset()
        action = self.get_action(state)
        done = False

        while not done:
            self.env.render()
            time.sleep(0.02)
            state, reward, done, info = self.env.step(action)
            action = self.get_action(state)

    def save(self, checkpoint_dir='checkpoint'):
        self.saver.save(self.sess, checkpoint_dir+'/'+self.name+'/model.ckpt', global_step=self.global_step)
        print('save 성공!')

    def load(self, checkpoint_dir='checkpoint'):
        '''
        self.saver = tf.train.Saver(var_list= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='main')\
            +tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target'))
        '''
        self.saver = tf.train.Saver(var_list= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

        if not os.path.isdir(checkpoint_dir+'/'+self.name):
            os.makedirs(checkpoint_dir+'/'+self.name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir+'/'+self.name)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print('success to load model!')

            self.epsilon = 0.3
        else:
            self.sess.run(tf.global_variables_initializer())
            print('fail to load model...')
        
    @staticmethod
    def _arg_max(state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)
