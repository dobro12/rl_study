from gym.envs.registration import register

import os
import os.path as osp
import subprocess

os.environ['QT_PLUGIN_PATH'] = osp.join(osp.dirname(osp.abspath(__file__)), '.qt_plugins') + ':' + \
                               os.environ.get('QT_PLUGIN_PATH','')

register(
    id='DobroQuadruped-v0',           #id is used for "gym.make("ID name")"
    entry_point='dobroEnv2:Env', #entry_point should be registered at PYTHONPATH
    max_episode_steps=1000
    )

from dobroEnv2.env import Env
