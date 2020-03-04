from dobroEnv.halfCheetah import HalfCheetah

from pybullet_utils import bullet_client
import pybullet_data
import pybullet as p1

import numpy as np
import random
import time
import copy
import gym
import os

class Env(gym.Env):
    action_dim = 6
    state_dim = 20
    action_space = gym.spaces.Box(-np.ones(action_dim), np.ones(action_dim), dtype=np.float32)
    observation_space = gym.spaces.Box(-np.inf*np.ones(state_dim), np.inf*np.ones(state_dim), dtype=np.float32)

    def __init__(self):
        self.time_step = 1. / 100.
        self.num_solver_iterations = 30
        self.sub_step = 1 #5
        self.elapsed_t = 0
        self.is_render = None
        self.not_move = 0
        self.smoothed_vel = 0

        self.time_step /= self.sub_step
        self.num_solver_iterations //= self.sub_step

    def initialize(self, is_render=False):
        self.is_render = is_render
        if self.is_render:
            self.pybullet_client = bullet_client.BulletClient(connection_mode=p1.GUI)
            self.pybullet_client.configureDebugVisualizer(self.pybullet_client.COV_ENABLE_GUI, 0)
        else:
            self.pybullet_client = bullet_client.BulletClient()
        self.pybullet_client.configureDebugVisualizer(self.pybullet_client.COV_ENABLE_Y_AXIS_UP, 1)
        self.pybullet_client.setAdditionalSearchPath('{}/urdf'.format(os.path.dirname(os.path.realpath(__file__))))
        self.pybullet_client.setPhysicsEngineParameter(numSolverIterations=self.num_solver_iterations) #10
        self.pybullet_client.setTimeStep(self.time_step)

        #make vertical plane(bottom plane)
        z2y = self.pybullet_client.getQuaternionFromEuler([-np.pi * 0.5, 0, 0])
        planeId = self.pybullet_client.loadURDF("plane_implicit.urdf", [0, 0, 0],
                                            z2y,
                                            useMaximalCoordinates=True)
        self.pybullet_client.setGravity(0, -9.8, 0)
        self.pybullet_client.changeDynamics(planeId, linkIndex=-1, lateralFriction=1.0)

        self.model = HalfCheetah(self.pybullet_client)

    def step(self, action):
        for i in range(self.sub_step):
            self.model.apply_action(action)
            self.pybullet_client.stepSimulation()
            self.elapsed_t += self.time_step
            if self.is_render:
                time.sleep(self.time_step)

        #camera
        if self.is_render:
            self.camera_move()

        base_vel = self.model.base_vel
        joint_vel = self.model.joint_vel
        state = self.model.get_state()

        '''
        w = 0.01
        self.smoothed_vel = self.smoothed_vel*(1-w) + base_vel[0]*w
        if self.smoothed_vel < 0.2:
            self.not_move += 1
        else:
            self.not_move = 0
        done = False
        if self.not_move > 100 or is_contact:
            done = True
        '''
        done = False
        if self.model.body_contact == 1.0:
            done = True

        electrocity_cost = np.abs(np.multiply(self.model.applied_forces, joint_vel)).mean()
        reward = base_vel[0] - 5e-4*electrocity_cost
        #reward = base_vel[0]
        info = None

        return state, reward, done, info

    def reset(self):
        self.elapsed_t = 0
        self.not_move = 0
        self.smoothed_vel = 0
        state = self.model.reset()

        #camera
        if self.is_render:
            self.camera_reset(self.model)
            self.camera_move()

        return state

    def render(self, mode='human'):
        pass
   
    def close(self):
        self.pybullet_client.disconnect()

    def camera_reset(self, target):
        self.camDist = 2.0
        self.camYaw = 180.0
        self.camPitch = -30.0
        self.camTarget = target
        if self.camTarget.base_idx == -1:
            camTargetPos, _ = self.pybullet_client.getBasePositionAndOrientation(self.camTarget.sim_model)
        else:
            camTargetPos, _, _, _, _, _ = self.pybullet_client.getLinkState(self.camTarget.sim_model, self.camTarget.base_idx)
        self.camTargetPos = np.array(camTargetPos)

    def camera_move(self, alpha = 0.9):
        if self.camTarget.base_idx == -1:
            camTargetPos, _ = self.pybullet_client.getBasePositionAndOrientation(self.camTarget.sim_model)
        else:
            camTargetPos, _, _, _, _, _ = self.pybullet_client.getLinkState(self.camTarget.sim_model, self.camTarget.base_idx)
        self.camTargetPos = alpha*self.camTargetPos + (1-alpha)*np.array(camTargetPos)
        self.pybullet_client.resetDebugVisualizerCamera(self.camDist, self.camYaw, self.camPitch, self.camTargetPos)
