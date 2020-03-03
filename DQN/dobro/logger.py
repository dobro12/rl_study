import time
import os
import pickle
import glob

class Logger:
    def __init__(self, env_name, save_name):
        now = time.localtime()
        save_name = '{}/{}_log'.format(env_name, save_name.lower())

        if not os.path.isdir(save_name):
            os.makedirs(save_name)
        self.log_name = save_name+"/record_%04d%02d%02d_%02d%02d%02d.pkl" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
        self.log = []


    def write(self, data):
        self.log.append(data)


    def save(self):
        with open(self.log_name, 'wb') as f:
            pickle.dump(self.log, f)
