import roboschool
import gym

from collections import deque
import numpy as np
import pickle
import tensorflow as tf
import tflearn
import random
import copy
import time
import os

from graph_drawer import Graph
from replay_buffer import ReplayBuffer

class OrnsteinUhlenbeckActionNoise:
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
    def __init__(self, env):
        self.name = 'DDPG'
        self.critic_lr = 1e-3
        self.actor_lr = 1e-4
        self.discount_factor = 0.99
        self.n_action = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]
        self.state_dim = env.observation_space.shape[0]
        self.env = env
        self.save_freq = 50
        self.soft_update = 1e-3

        #self.replay_memory = deque(maxlen=2000)
        try:
            with open('replay.pkl', 'rb') as f:
                self.replay_memory = pickle.load(f)
        except:
            self.replay_memory = ReplayBuffer(100000, 1234)
        self.train_start = int(1e4)
        self.batch_size = int(1e3)#256
        self.epsilon = 1.0
        self.epsilon_decay = 0.9999
        self.min_epsilon = 0.01

        self.actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.n_action))

        self.X = tf.placeholder(tf.float32, [None, self.state_dim], name='X')
        self.Q_Action_Gradient = tf.placeholder(tf.float32, [None, self.n_action], name='Q_Action_Gradient')
        self.A = tf.placeholder(tf.float32, [None, self.n_action], name='Action')
        self.Y = tf.placeholder(tf.float32, [None, 1], name='Target_Q_value')

        with tf.variable_scope('critic'):
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.critic = self._build_critic()
        self.target_critic = self._build_critic('target_critic')
        self.train_critic_op, self.Q_loss = self._build_critic_op()
        self.actor = self._build_actor()
        self.target_actor = self._build_actor('target_actor')
        self.train_actor_op = self._build_actor_op()
        self.update_target_model = self._build_update_target_model()
        self.action_q_gradient = tf.gradients(self.critic, self.A)

        self.sess = tf.Session()
        self.load()
        self.init_target_model()

        print(self.sess.run(self.global_step))

    def _build_actor(self, name='actor'):
        with tf.variable_scope(name):
            model = tf.layers.dense(self.X, 300, name='dense_0', activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            model = tf.layers.batch_normalization(model)
            model = tf.nn.relu(model)

            model = tf.layers.dense(model, 600, name='dense_1', activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            model = tf.layers.batch_normalization(model)
            model = tf.nn.relu(model)

            model = tf.layers.dense(model, self.n_action, name='dense_2', activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            model = tf.nn.tanh(model)
            policy = tf.multiply(model, self.action_bound)

            return policy

    def _build_actor_op(self, name='actor'):
        model_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        gradients = tf.gradients(self.actor, model_vars, -self.Q_Action_Gradient)
        batch_gradients = list(map(lambda x: tf.div(x, self.batch_size), gradients))

        with tf.variable_scope(name):
            train_op = tf.train.AdamOptimizer(self.actor_lr, beta1=0.5, beta2=0.999).apply_gradients(zip(batch_gradients, model_vars))

        return train_op

    def _build_critic(self, name='critic'):
        with tf.variable_scope(name):
            model = tf.layers.dense(self.X, 400, name='dense_0', activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            model = tf.layers.batch_normalization(model)
            model = tf.nn.relu(model)

            t1 = tf.layers.dense(model, 300, name='dense_1', use_bias=False, activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            t2 = tf.layers.dense(self.A, 300, name='dense_2', activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))

            model = tf.nn.relu(tf.add(t1, t2))

            Q = tf.layers.dense(model, 1, name='dense_3', activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            return Q

    def _build_critic_op(self, name='critic'):
        cost = tf.reduce_mean(tf.square(self.Y - self.critic))

        model_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        with tf.variable_scope(name):
            train_op = tf.train.AdamOptimizer(self.critic_lr, beta1=0.5, beta2=0.999).minimize(cost, var_list=model_vars, global_step=self.global_step)

        return train_op, cost

    def _build_update_target_model(self):
        copy_op = []

        main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_actor')

        # 학습 네트웍의 변수의 값들을 타겟 네트웍으로 복사해서 타겟 네트웍의 값들을 최신으로 업데이트합니다.
        for main_var, target_var in zip(main_vars, target_vars):
            copy_op.append(target_var.assign( tf.multiply( main_var.value(), self.soft_update) + tf.multiply( target_var.value(), 1.0 - self.soft_update) ))

        main_vars2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
        target_vars2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_critic')

        # 학습 네트웍의 변수의 값들을 타겟 네트웍으로 복사해서 타겟 네트웍의 값들을 최신으로 업데이트합니다.
        for main_var, target_var in zip(main_vars2, target_vars2):
            copy_op.append(target_var.assign( tf.multiply( main_var.value(), self.soft_update) + tf.multiply( target_var.value(), 1.0 - self.soft_update) ))

        return copy_op

    def init_target_model(self):
        copy_op = []

        main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_actor')

        # 학습 네트웍의 변수의 값들을 타겟 네트웍으로 복사해서 타겟 네트웍의 값들을 최신으로 업데이트합니다.
        for main_var, target_var in zip(main_vars, target_vars):
            copy_op.append(target_var.assign(main_var.value()))

        main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_critic')

        # 학습 네트웍의 변수의 값들을 타겟 네트웍으로 복사해서 타겟 네트웍의 값들을 최신으로 업데이트합니다.
        for main_var, target_var in zip(main_vars, target_vars):
            copy_op.append(target_var.assign(main_var.value()))

        self.sess.run(copy_op)

    def get_action(self, state):
        [actions] = self.sess.run(self.actor, feed_dict={self.X:[state]})
        for i in range(len(actions)):
            actions[i] += np.random.normal(scale=self.epsilon)
        actions = np.clip(actions, -self.action_bound, self.action_bound)
        return actions

    def _train(self):
        '''
        mini_batch = random.sample(self.replay_memory, self.batch_size)

        states = []
        actions = []
        rewards = []
        dones = []
        next_states = []
        for batch in mini_batch:
            states.append(batch[0])
            actions.append(batch[1])
            rewards.append(batch[2])
            dones.append(batch[3])
            next_states.append(batch[4])
        '''
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
        states, actions, rewards, dones, next_states = self.replay_memory.sample_batch(self.batch_size)

        next_actions = self.sess.run(self.target_actor, feed_dict={self.X:next_states})
        next_Q_value = self.sess.run(self.target_critic, feed_dict={self.X:next_states, self.A:next_actions})

        critic_target = []
        for i in range(self.batch_size):
            if dones[i]:
                critic_target.append([rewards[i]])
            else:
                critic_target.append([rewards[i] + self.discount_factor*next_Q_value[i][0]])
        Q_loss,_ = self.sess.run([self.Q_loss, self.train_critic_op], feed_dict={self.X:states, self.A :actions, self.Y:critic_target})

        mu_actions = self.sess.run(self.actor, feed_dict={self.X:states})
        [action_gradient] = self.sess.run(self.action_q_gradient, feed_dict={self.X:states, self.A:mu_actions})
        self.sess.run(self.train_actor_op, feed_dict={self.X:states, self.Q_Action_Gradient:action_gradient})

        return Q_loss

    def train(self, episodes=10000):
        graph = Graph(freq=1000, title='HOPPER', label='DDPG')

        for episode in range(episodes):
            state = self.env.reset()
            score = 0
            done = False

            Q_loss = 0
            count = 0

            while not done:
                action = self.get_action(state)# + self.actor_noise()
                next_state, reward, done, info = self.env.step(action)

                #self.replay_memory.append((state, action, reward, done, next_state))
                self.replay_memory.add(state, action, reward, done, next_state)

                #if len(self.replay_memory) > self.train_start:
                if self.replay_memory.size() > self.batch_size:
                    Q_loss += self._train()
                    self.sess.run(self.update_target_model)
                    #print(reward)

                state = next_state
                score += reward
                count += 1

            #draw graph
            graph.update(score, Q_loss/count)

            print("ep : {} | score : {} | epsilon : {} | Q_loss : {:.3f}".format(episode, score, self.epsilon, Q_loss/count))
            if (episode+1)%self.save_freq == 0:
                self.save()
                with open('replay.pkl', 'wb') as f:
                    pickle.dump(self.replay_memory, f)
        
    def test(self):
        state = self.env.reset()
        score = 0
        done = False

        ep_ave_max_q = 0
        count = 0

        while not done:
            self.env.render()
            action = self.get_action(state)
            next_state, reward, done, info = self.env.step(action)

            time.sleep(0.02)
            print(next_state[1])

            state = next_state
            score += reward
            count += 1

        print(score, ep_ave_max_q/float(count))

    def save(self, checkpoint_dir='checkpoint'):
        self.saver.save(self.sess, checkpoint_dir+'/'+self.name+'/model.ckpt', global_step=self.global_step)
        print('save 성공!')

    def load(self, checkpoint_dir='checkpoint'):
        '''
        self.saver = tf.train.Saver(var_list= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor')\
            +tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic'))
        '''
        self.saver = tf.train.Saver(var_list= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

        if not os.path.isdir(checkpoint_dir+'/'+self.name):
            os.makedirs(checkpoint_dir+'/'+self.name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir+'/'+self.name)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print('success to load model!')

            self.epsilon = self.min_epsilon
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


if __name__ == '__main__':
    #env = gym.make('HalfCheetah-v2')
    #env = gym.make('Hopper-v2')
    #env = gym.make('RoboschoolHopper-v1')
    env = gym.make('RoboschoolHalfCheetah-v1')
    agent = Agent(env)
    #agent.train()
    agent.test()
