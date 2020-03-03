from collections import deque
import tensorflow as tf
import numpy as np
import random
import copy
import time
import os

class Agent:
    def __init__(self, env, env_name):
        self.env = env
        self.name = 'dqn'
        self.checkpoint_dir='{}/checkpoint'.format(env_name)
        self.epsilon = 1.0
        self.epsilon_decay = 0.9999
        self.min_epsilon = 0.01
        self.learning_rate = 1e-3
        self.discount_factor = 0.99
        self.n_action = 3
        self.state_dim = env.observation_space.shape[0]
        self.save_freq = 10

        # 리플레이 메모리, 최대 크기 2000
        self.replay_memory = deque(maxlen=2000)
        self.target_upadate_freq = 1000
        self.train_start = 1000
        self.batch_size = 64

        self.X = tf.placeholder(tf.float32, [None, self.state_dim], name='X')
        self.A = tf.placeholder(tf.int32, [None], name='Action')
        self.Y = tf.placeholder(tf.float32, [None], name='Y')

        self.Q = self._build_model('main')
        self.train_op, self.loss = self._build_op('main')
        self.target_Q = self._build_model('target')
        self.copy_op = self._build_update_target_model()

        self.sess = tf.Session()
        self.load()
        self.update_target_model()


    def _build_model(self, name):
        with tf.variable_scope(name):
            model = tf.layers.dense(self.X, 128, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            model = tf.layers.dense(model, 128, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            model = tf.layers.dense(model, 128, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            Q = tf.layers.dense(model, self.n_action, activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
        return Q

    def _build_op(self, name):
        one_hot = tf.one_hot(self.A, self.n_action, 1.0, 0.0)
        Q_value = tf.reduce_sum(tf.multiply(one_hot, self.Q), axis=1)
        loss = tf.reduce_mean(tf.square(self.Y - Q_value))

        model_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        train_op = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5, beta2=0.999).minimize(loss, var_list=model_vars)
        return train_op, loss

    def _build_update_target_model(self):
        copy_op = []

        main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='main')
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')

        # 학습 네트웍의 변수의 값들을 타겟 네트웍으로 복사해서 타겟 네트웍의 값들을 최신으로 업데이트합니다.
        for main_var, target_var in zip(main_vars, target_vars):
            copy_op.append(target_var.assign(main_var.value()))

        return copy_op

    def update_target_model(self):
        self.sess.run(self.copy_op)

    def get_action(self, state):
        rand_number = random.random()
        if rand_number < self.epsilon:
            #action = random.choice(self.actions)
            action = random.randint(0, self.n_action-1)
        else:
            [Q_value] = self.sess.run(self.Q, feed_dict={self.X:[state]})
            action = np.argmax(Q_value)
        return action

    def train(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay

        mini_batch = random.sample(self.replay_memory, self.batch_size)
        next_state = [batch[4] for batch in mini_batch]
        next_Q_value = self.sess.run(self.target_Q, feed_dict={self.X:next_state})

        _target = []
        _state = []
        _action = []
        for i, [state, action, reward, done, next_state] in enumerate(mini_batch):
            _state.append(state)
            _action.append(action)
            if done:
                _target.append(reward)
            else:
                _target.append(reward + self.discount_factor*np.amax(next_Q_value[i]))
        _, Q, loss = self.sess.run([self.train_op, self.Q, self.loss], feed_dict={self.X:_state, self.Y:_target, self.A:_action})
        return Q, loss
        
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
