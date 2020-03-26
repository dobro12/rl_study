from gym.envs.registration import register

import os
import os.path as osp
import subprocess

os.environ['QT_PLUGIN_PATH'] = osp.join(osp.dirname(osp.abspath(__file__)), '.qt_plugins') + ':' + \
                               os.environ.get('QT_PLUGIN_PATH','')

register(
    id='DobroHalfCheetah-v0',           #id is used for "gym.make("ID name")"
    entry_point='dobroEnv:Env', #entry_point should be registered at PYTHONPATH
    max_episode_steps=1000
    )

from dobroEnv.env import Env