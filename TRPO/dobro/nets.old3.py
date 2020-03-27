from sklearn.utils import shuffle
from collections import deque
from copy import deepcopy
import tensorflow as tf
import numpy as np
import itertools
import pickle
import random
import copy
import time
import os

class Agent:
    def __init__(self, env, args):
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
        self.batch_size = args['batch_size']

        self.value_epochs = 10
        self.num_conjugate = 10
        self.delta = 0.01
        self.line_decay = 0.5
        self.max_decay_num = 100

        with tf.variable_scope(self.name):
            #placeholder
            self.states = tf.placeholder(tf.float32, [None, self.state_dim], name='State')
            self.actions = tf.placeholder(tf.float32, [None, self.action_dim], name='Action')
            self.targets = tf.placeholder(tf.float32, [None,], name='targets')
            self.old_mean = tf.placeholder(tf.float32, [None, self.action_dim], name='old_mean')

            #policy & value
            self.mean = self.build_policy_model('policy')
            self.std = args['std']
            self.value = self.build_value_model('value')

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
            self.v_gradients = tf.gradients(self.v_loss, v_vars)
            self.v_train_op = v_optimizer.apply_gradients(zip(self.v_gradients, v_vars))

            #policy optimizer
            norm_actions = self.normalize_action(self.actions)
            p_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name+'/policy')
            log_objective = - tf.reduce_sum(np.log(self.std) + 0.5*np.log(2*np.pi) + tf.squared_difference(norm_actions, self.mean) / (2 * self.std**2), axis=1)
            log_objective = tf.reduce_mean(log_objective * (self.targets - self.value))
            #log_objective = tf.reduce_mean(log_objective * self.targets)
            objective = (tf.squared_difference(self.old_mean, norm_actions) - tf.squared_difference(self.mean, norm_actions))/(2*self.std**2)
            self.objective = tf.reduce_mean(tf.exp(tf.reduce_sum(objective, axis=1)) * (self.targets - self.value))
            #self.objective = tf.reduce_mean(tf.exp(tf.reduce_sum(objective, axis=1)) * self.targets)
            self.g = tf.gradients(log_objective, p_vars)
            self.g = tf.concat([tf.reshape(g, [-1]) for g in self.g], axis=0)

            J = []
            mean_list = tf.split(self.mean, [1 for i in range(self.action_dim)], axis=1)
            for ith_mean in mean_list:
                ith_J = tf.gradients(tf.reduce_mean(ith_mean), p_vars)
                ith_J = tf.concat([tf.reshape(g, [-1]) for g in ith_J], axis=0)
                J.append(ith_J)
            self.J = tf.stack(J, axis=0)
            self.A = tf.matmul(tf.transpose(self.J), self.J) / (self.std**2) 

            kl = tf.reduce_sum(0.5*tf.square((self.mean - self.old_mean)/self.std),axis=1)
            self.kl = tf.reduce_mean(kl)

            self.flatten_p_vars = tf.concat([tf.reshape(g, [-1]) for g in p_vars], axis=0)
            self.params = tf.placeholder(tf.float32, self.flatten_p_vars.shape, name='params')
            self.assign_op = []
            start = 0
            for p_var in p_vars:
                size = np.prod(p_var.shape)
                param = tf.reshape(self.params[start:start + size], p_var.shape)
                self.assign_op.append(p_var.assign(param))
                start += size

            #make session and load model
            self.sess = tf.Session()
            self.load()


    def build_policy_model(self, name='policy'):
        with tf.variable_scope(name):
            model = tf.layers.dense(self.states, self.hidden1_units, activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            #model = tf.layers.batch_normalization(model)
            model = tf.nn.tanh(model)
            model = tf.layers.dense(model, self.hidden2_units, activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            #model = tf.layers.batch_normalization(model)
            model = tf.nn.tanh(model)
            model = tf.layers.dense(model, self.action_dim, activation=tf.tanh, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
        return model

    def build_value_model(self, name='value'):
        with tf.variable_scope(name):
            inputs = self.states
            model = tf.layers.dense(inputs, self.hidden1_units, activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            #model = tf.layers.batch_normalization(model)
            model = tf.nn.tanh(model)
            model = tf.layers.dense(model, self.hidden2_units, activation=None, kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
            #model = tf.layers.batch_normalization(model)
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

    def get_action(self, state, is_train):
        if is_train:
            [action] = self.sess.run(self.sample_noise_action, feed_dict={self.states:[state]})
        else:
            [action] = self.sess.run(self.sample_action, feed_dict={self.states:[state]})
        #action = np.clip(action, self.action_bound_min, self.action_bound_max)
        return action

    def train(self, trajs):
        states = trajs[0]
        actions = trajs[1]
        rewards = trajs[2]
        next_states = trajs[3]
        dones = trajs[4]
        targets = trajs[5]
        old_means = self.sess.run(self.mean, feed_dict={self.states:states})

        #POLICY update
        A, g = self.sess.run([self.A, self.g], feed_dict={
            self.states:states,
            self.actions:actions,
            self.targets:targets})
        #for i, cond in enumerate(abs(J[0]) < 0.01):
        #    if cond: J[0,i] = 0.0
        #A = np.dot(J.T, J)

        '''
        strs = "["
        for i in range(A.shape[0]):
            strs+="["
            for j in range(A.shape[1]):
                strs+="{}, ".format(A[i,j])
            strs = strs[:-2]
            strs+="],"
        strs = strs[:-1]
        strs+="]"
        print(strs)
        strs = "["
        for i in range(J.shape[1]):
            strs+="{}, ".format(J[0, i])
        strs = strs[:-2]
        strs+="]"
        print(strs)
        '''

        approxA = A + 1e-2 #실제로 A는 역행렬이 존재하는데, 아이젠벨류가 너무 작아 안되는걸 막아줌. 매우작은 항등행렬을 추가해주는 효과!
        x_value = self.conjugate_gradient_method(approxA, g)

        '''
        while True:
            P = np.random.normal(size=A.shape)*0.001
            A += (P.T + P)
            x_value = np.matmul(np.linalg.inv(A), g)
            if not np.isnan(np.sum(x_value)):
                break
        '''
        #residue = np.matmul(A, x_value) - g
        #rs = np.inner(residue, residue)
        #print("residue :",np.sqrt(rs), np.sqrt(np.inner(g,g)), np.sqrt(np.inner(x_value,x_value)))

        #line search
        #xAx = np.inner(x_value, np.matmul(A, x_value))
        xAx = np.clip(np.inner(x_value, np.matmul(A, x_value)), 1e-2, np.inf)
        beta = np.sqrt(2*self.delta / xAx)
        if np.isnan(beta):
            print(x_value)
            print(np.inner(x_value, np.matmul(A, x_value)))

            x_value = np.matmul(np.linalg.inv(A), g)
            residue = np.matmul(A, x_value) - g
            rs = np.inner(residue, residue)
            print("real residue :", np.sqrt(rs))

            x_value = np.matmul(np.linalg.inv(approxA), g)
            residue = np.matmul(A, x_value) - g
            rs = np.inner(residue, residue)
            print("approx residue :", np.sqrt(rs))

            raise ValueError("beta is NaN!")
        init_theta = self.sess.run(self.flatten_p_vars)
        max_objective = None
        max_theta = None
        max_kl = None
        cnt = 0

        #while True:
        for i in range(self.max_decay_num):
            cnt += 1
            theta = beta*x_value + init_theta
            self.sess.run(self.assign_op, feed_dict={self.params:theta})
            kl, objective = self.sess.run([self.kl, self.objective], feed_dict={
                self.states:states,
                self.actions:actions,
                self.targets:targets,
                self.old_mean:old_means})
            if max_objective == None:
                if kl <= self.delta:
                    max_objective = objective
                    max_theta = deepcopy(theta)
                    max_kl = kl
            else:
                #if kl > self.delta:
                #    break
                #if max_objective > objective + 0.1:
                #    break
                if max_objective < objective and kl <= self.delta:
                    max_objective = objective                    
                    max_theta = deepcopy(theta)
                    max_kl = kl
            #if cnt > self.max_decay_num and max_objective != None:
            #    break
            beta *= self.line_decay
        #if max_objective == None:
        #    max_objective = objective
        #    max_theta = deepcopy(theta)
        #    max_kl = kl
        self.sess.run(self.assign_op, feed_dict={self.params:max_theta})

        #VALUE update
        num_batches = len(states) // self.batch_size
        v_s, v_t = shuffle(states, targets, random_state=0)
        for _ in range(self.value_epochs):
            v_s, v_t = shuffle(v_s, v_t, random_state=0)
            self.sess.run(self.v_train_op, feed_dict={
                    self.states:v_s, 
                    self.targets:v_t})

        v_loss = self.sess.run(self.v_loss, feed_dict={self.states:states, self.targets:targets})
        return v_loss, max_objective, max_kl


    def conjugate_gradient_method(self, A, g):
        x_value = np.zeros_like(g)
        residue = deepcopy(g)
        p_vector = deepcopy(g)
        rs_old = np.inner(residue, residue)

        for i in range(self.num_conjugate):
            Ap = np.matmul(A, p_vector)
            pAp = np.inner(p_vector, Ap)
            alpha = rs_old / pAp
            x_value += alpha * p_vector
            residue -= alpha * Ap
            rs_new = np.inner(residue, residue)

            if np.sqrt(rs_new) < 1e-5:
                break
            p_vector = residue + (rs_new / (1e-10 + rs_old)) * p_vector
            rs_old = rs_new

        return x_value

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
        

if __name__ == "__main__":
    env_name = 'Pendulum-v0'
    save_name = env_name.split('-')[0]
    agent_args = {'agent_name':'TRPO',
                'env_name':save_name,
                'discount_factor':0.9,
                'hidden1':2,
                'hidden2':2,
                'v_lr':1e-3,
                'batch_size':128,
                'std':0.1}

    import gym
    env = gym.make(env_name)
    agent = Agent(env, agent_args)

    states = []
    actions = []
    rewards = []
    next_states = []
    state = env.reset()
    done = False
    while not done:
        action = agent.get_action(state, True)
        next_state, reward, done, info = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        
        state = next_state

    trajs = [states, actions, rewards, next_states]
    agent.train(trajs)

