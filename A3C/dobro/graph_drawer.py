import matplotlib.pyplot as plt
from collections import deque
import multiprocessing as mp
import numpy as np
import time

class ProcessPlotter(object):
    def __init__(self, freq, title, label):
        max_len = int(5e3)
        self.x = deque(maxlen=max_len)
        self.y1 = deque(maxlen=max_len)
        self.y2 = deque(maxlen=max_len)
        self.y3_0 = deque(maxlen=max_len)
        self.y3_1 = deque(maxlen=max_len)
        self.y4 = deque(maxlen=max_len)
        self.interval = freq
        self.title = title
        self.label = label

    def terminate(self):
        plt.close('all')

    def call_back(self):
        while self.pipe.poll():
            command = self.pipe.recv()
            if command is None:
                self.terminate()
                return False
            else:
                self.x.append(command[0])
                self.y1.append(command[1])
                self.y2.append(command[2])
                self.y3_0.append(command[3][0])
                self.y3_1.append(command[3][1])
                self.y4.append(command[4])
        del self.ax.lines[-1]
        del self.ax2.lines[-1]
        for i in range(len(self.ax3.lines)):
            del self.ax3.lines[0]
        del self.ax4.lines[-1]
        if len(self.x) == 0:
            lower, upper = 0, 0
        else:
            lower, upper = self.x[0], self.x[-1]
        self.ax.plot(self.x, self.y1, 'r')
        self.ax.set_xlim(lower, upper)
        self.ax2.plot(self.x, self.y2, 'b')
        self.ax2.set_xlim(lower, upper)
        self.ax3.plot(self.x, self.y3_0, 'g')
        self.ax3.plot(self.x, self.y3_1, 'black')
        self.ax3.set_xlim(lower, upper)
        self.ax4.plot(self.x, self.y4, 'y')
        self.ax4.set_xlim(lower, upper)

        self.fig.canvas.draw()
        return True

    def __call__(self, pipe):
        print('starting plotter...')

        self.pipe = pipe
        #self.fig, self.ax = plt.subplots()
        self.fig = plt.figure(figsize=(16,4))
        self.fig.suptitle('{} - {}'.format(self.title, self.label), fontsize=16)

        self.ax = self.fig.add_subplot(1,4,1)
        self.ax.plot([], [], 'r')
        self.ax.set_title('rewards')
        self.ax.set_xlabel('iters')
        self.ax.set_ylabel('')
        plt.grid()

        self.ax2 = self.fig.add_subplot(1,4,2)
        self.ax2.plot([], [], 'b')
        self.ax2.set_title('policy loss')
        self.ax2.set_xlabel('iters')
        self.ax2.set_ylabel('')
        plt.grid()

        self.ax3 = self.fig.add_subplot(1,4,3)
        self.ax3.plot([], [], 'g')
        self.ax3.set_title('value loss')
        self.ax3.set_xlabel('iters')
        self.ax3.set_ylabel('')
        plt.grid()

        self.ax4 = self.fig.add_subplot(1,4,4)
        self.ax4.plot([], [], 'y')
        self.ax4.set_title('entropy')
        self.ax4.set_xlabel('iters')
        self.ax4.set_ylabel('')
        plt.grid()

        timer = self.fig.canvas.new_timer(interval=self.interval)
        timer.add_callback(self.call_back)
        timer.start()

        print('...done')
        plt.show()

class Graph:
    def __init__(self, freq, title, label):
        self.plot_pipe, plotter_pipe = mp.Pipe()
        self.plotter = ProcessPlotter(freq=freq, title=title, label=label)
        self.plot_process = mp.Process(
            target=self.plotter, args=(plotter_pipe,), daemon=True)
        self.plot_process.start()

        self.count = 0

    def update(self, reward, loss_p, loss_v, entropy, finished=False):
        self.count += 1

        send = self.plot_pipe.send
        if finished:
            send(None)
        else:
            send([self.count, reward, loss_p, loss_v, entropy])
