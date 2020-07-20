import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='plot_reward_graph')
parser.add_argument('-graph', default='average' , help='type of graph' )
parser.add_argument('-path', help='path(relative) of data')
parser.add_argument('-win_size', default=10 ,type=int , help='window size for smoothing')
args = parser.parse_args()

path = os.path.join(os.getcwd() , args.path)

with open(path,'rb') as f:
    data = pickle.load(f)

data = np.array(data)

mean_data = np.mean(data,axis = 0)
std_data = np.var(data, axis=0) ** (1/2.)

def plot_average():
    fig, ax = plt.subplots(1)
    ax.plot(mean_data , lw=2 , label = 'avaerage_reward', color='blue')
    ax.fill_between(np.arange(len(mean_data)), mean_data + std_data, mean_data - std_data , facecolor = 'blue', alpha=0.5)
    ax.set_title('average_reward')
    ax.legend(loc='upper left')
    ax.set_xlabel('num episode')
    ax.set_ylabel('reward')
    ax.grid()

    plt.show()

def plot_smoothed(window_size):
    smoothed_mean = np.zeros_like(mean_data)
    smoothed_std = np.zeros_like(std_data)
    for i in range(data.shape[1]):
        smoothed_mean[i] = mean_data[0 if i<9 else i-9  : i+1].mean()
        smoothed_std[i] = std_data[0 if i<9 else i-9 : i+1].mean()

    fig, ax = plt.subplots(1)
    ax.plot(smoothed_mean , lw=2 , label = 'smoothed_reward', color='blue')
    ax.fill_between(np.arange(len(mean_data)), smoothed_mean + smoothed_std, smoothed_mean - smoothed_std , facecolor = 'blue', alpha=0.5)
    ax.set_title('smoothed_reward')
    ax.legend(loc='upper left')
    ax.set_xlabel('num episode')
    ax.set_ylabel('reward')
    ax.grid()

    plt.show()

if(args.graph == 'average'):
    plot_average()
elif(args.graph == 'smoothed'):
    plot_smoothed(args.win_size)
else:
    print('error')



