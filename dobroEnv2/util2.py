import tensorflow as tf
import numpy as np
import random
import time

def random_seed(seed):
    np.random.seed(seed)
    tf.set_random_seed(seed)
    random.seed(seed)

def get_reward(r_t, goal):
    #r_t : alive, vel_x, electricity_cost, joints_at_limit_cost, feet_collision_cost
    #r_t = np.sum(r_t) #at roboschool env, we didn't check this reward
    #r_t = r_t[1] + r_t[2] #at mujoco env, and it did very well
    '''
    goal_vel = goal[0]
    goal_cost = 10*np.exp(-np.abs(r_t[1] - goal_vel))
    #r_t = r_t[0] + goal_cost + r_t[2] + r_t[3] + r_t[4]
    #r_t = goal_cost + r_t[2]
    '''
    #reward = [base_vel[2], alive_bonus, y_move_penalty, horizon_penalty, electrocitiy_cost, stall_cost]

    goal_height = goal[0]
    r_t = -abs(r_t[0]) + r_t[1] - abs(r_t[2] - goal_height)*0.1 + r_t[3]
    #r_t = -abs(r_t[0]) + r_t[1] - abs(r_t[2] - goal_height)*0.1 + r_t[3] - r_t[4]*0.001 - r_t[5]*0.0001
    #r_t = r_t[0] + r_t[1] + r_t[3] - r_t[4]*0.1 - r_t[5]
    #print(r_t)
    return r_t

def run_episodes(env, agent, episode, is_train, is_root, maxstep):
    print("##### {} Episode start! #####".format(episode))
    step = 0
    episodes = 0
    trajectories = []
    while step < maxstep:
        #print("##### {} Episode start! #####".format(episode))
        episode += 1
        episodes += 1
        states = []
        goals = []
        actions = []
        values = []
        true_rewards = []
        targets = []
        stds = []
        s_t = env.reset()
        start_t = time.time()
        g_t = [float(np.random.randint(20,50))/100.0]
        #g_t = [0.5]
        #g_t = [0.0]
        if (not is_train) and is_root:
            print('goal :',g_t)
        while True :
            #a_t, value = agent.get_action(s_t, g_t, is_train)
            a_t, value, std = agent.get_action(s_t, g_t, True)
            '''
            ########################################
            joint_state = s_t[10:22]
            joint_vel = s_t[22:34]
            action = np.zeros_like(a_t)
            Kp = 1.0
            Kd = 0.015
            for i in range(len(action)):
                action[i] = Kp*(a_t[i] - joint_state[i]) - Kd*joint_vel[i]
            s_t1, r_t, done, info = env.step(action)
            ########################################
            '''
            s_t1, r_t, done, info = env.step(a_t)
            targets.append(r_t[2])
            r_t = get_reward(r_t, g_t)

            if (not is_train) and is_root:
                elapsed_t = time.time() - start_t
                remained_t = max(env.time_step*env.sub_step - elapsed_t, 0)
                time.sleep(remained_t)
                #time.sleep(0.1)
                #print(a_t)
                start_t = time.time()
            states.append(s_t)
            goals.append(g_t)
            actions.append(a_t)
            values.append(value)
            true_rewards.append(r_t)
            stds.append(std)
            step += 1
            if done:
                break
            s_t = s_t1

        if (not is_train) and is_root: 
            #print(np.mean(actions, axis=0))
            #print(np.var(actions, axis=0))
            print(np.sum(true_rewards), '| avg target :', np.mean(targets))
            time.sleep(0.5)
        next_values = values[1:]
        next_values.append(0)
        states = np.array(states).astype(dtype=np.float32)
        goals = np.array(goals).astype(dtype=np.float32)
        actions = np.array(actions).astype(dtype=np.float32)
        values = np.array(values).astype(dtype=np.float32)
        next_values = np.array(next_values).astype(dtype=np.float32)
        stds = np.array(stds).astype(dtype=np.float32)
        trajectories.append([states, actions, values, next_values, true_rewards, goals, stds])

    return episodes, trajectories
