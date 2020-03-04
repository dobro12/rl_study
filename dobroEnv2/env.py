from dobroEnv2.quadruped2 import Quadruped

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
  action_dim = 12
  state_dim = 38
  action_space = gym.spaces.Box(-np.ones(action_dim), np.ones(action_dim), dtype=np.float32)
  observation_space = gym.spaces.Box(-np.inf*np.ones(state_dim), np.inf*np.ones(state_dim), dtype=np.float32)

  def __init__(self):
    self.time_step = 1/240 #1. / 100.
    self.num_solver_iterations = 100 #30
    self.sub_step = 1 #5
    self.elapsed_t = 0
    self.is_render = None

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
    #self.pybullet_client.changeDynamics(planeId, linkIndex=-1, lateralFriction=0.9)
    self.pybullet_client.changeDynamics(planeId, linkIndex=-1, lateralFriction=1.0)
    #self.pybullet_client.changeDynamics(planeId, linkIndex=-1, lateralFriction=0.0)

    self.model = Quadruped(self.pybullet_client, useFixedBase=False)

  def step(self, action):
    for i in range(self.sub_step):
      self.model.apply_action(action)
      self.pybullet_client.stepSimulation()
      self.elapsed_t += self.time_step
      if self.is_render:
        time.sleep(self.time_step)
      #####################################################
      #self.model.pybullet_client.resetBasePositionAndOrientation(self.model.sim_model, self.model.init_base_pos, self.model.init_base_orn)
      #self.model.pybullet_client.resetBaseVelocity(self.model.sim_model, np.zeros(3), np.zeros(3))
      #####################################################

    # joint의 upper limit, lowwer limit이 구현이 안돼 수동구현함
    for i, joint in enumerate(self.model.joint_pos):
      if self.model.joint_info[i][8]>joint:
        self.pybullet_client.resetJointState(self.model.sim_model, self.model.joints[i], self.model.joint_info[i][8], 0.0)
      elif self.model.joint_info[i][9]<joint:
        self.pybullet_client.resetJointState(self.model.sim_model, self.model.joints[i], self.model.joint_info[i][9], 0.0)
    # joint의 upper limit, lowwer limit이 구현이 안돼 수동구현함

    #camera
    self.camera_move()

    state = self.model.get_state()
    base_vel = self.model.base_vel
    body_contact = self.model.body_contact
    base_org = self.pybullet_client.getEulerFromQuaternion(self.model.base_orn)

    done = False
    if body_contact or self.elapsed_t >= 10 or abs(base_org[0]) >= np.pi/3:
      done = True

    alive_bonus = 0 if done else 1
    #y_move_penalty = -base_vel[1]
    y_move_penalty = self.model.base_pos[1]
    horizon_penalty = -abs(base_org[0])
    joint_vel = self.model.joint_vel
    electrocity_cost = np.abs(np.multiply(self.model.applied_forces, joint_vel)).mean()
    stall_cost = np.square(self.model.applied_forces).mean()
    #print(electrocity_cost, stall_cost)
    #reward = base_vel[2] + alive_bonus*0.3 + y_move_penalty*0.5 + horizon_penalty*0.1
    r_t = [base_vel[2], alive_bonus, y_move_penalty, horizon_penalty, electrocity_cost, stall_cost]
    reward = r_t[0] + r_t[1] - r_t[4]*0.1 - r_t[5]

    info = None

    return state, reward, done, info

  def reset(self):
    self.elapsed_t = 0
    state = self.model.reset()

    #camera
    self.camera_reset(self.model.sim_model)
    self.camera_move()

    return state 

  def render(self, mode='human'):
    pass

  def close(self):
    self.pybullet_client.disconnect()

  def camera_reset(self, target):
    self.camDist = 2.0
    self.camYaw = 90.0 #0.0
    self.camPitch = -30.0
    self.camTarget = target
    self.camTargetPos, _ = self.pybullet_client.getBasePositionAndOrientation(self.camTarget)
    self.camTargetPos = np.array(self.camTargetPos)

  def camera_move(self, alpha = 0.9):
    targetPos, _ = self.pybullet_client.getBasePositionAndOrientation(self.camTarget)
    targetPos = np.array(targetPos)
    self.camTargetPos = alpha*self.camTargetPos + (1-alpha)*targetPos
    self.pybullet_client.resetDebugVisualizerCamera(self.camDist, self.camYaw, self.camPitch, self.camTargetPos)
