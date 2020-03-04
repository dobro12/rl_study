from pybullet_utils import bullet_client
import pybullet_data
import pybullet as p1

import numpy as np
import time
import random
import copy


def x_rot(t):
  rot = [[1.0,       0.0,        0.0],
          [0.0, np.cos(t), -np.sin(t)],
          [0.0, np.sin(t), np.cos(t)]]
  return np.array(rot)
def y_rot(t):
  rot = [[np.cos(t), 0.0, np.sin(t)],
          [      0.0, 1.0,       0.0],
        [-np.sin(t), 0.0, np.cos(t)]]
  return np.array(rot)
def z_rot(t):
  rot = [[np.cos(t), -np.sin(t), 0.0],
          [np.sin(t),  np.cos(t), 0.0],
          [      0.0,        0.0, 1.0]]
  return np.array(rot)


class Quadruped(object):
  def __init__( self, pybullet_client, useFixedBase=True, arg_parser=None):
    self.pybullet_client = pybullet_client

    #self.init_base_pos = [0, 0.55, 0]
    self.init_base_pos = [0, 0.6, 0]
    #self.init_base_pos = [0, 0.25, 0]
    self.init_base_orn = self.pybullet_client.getQuaternionFromEuler([0, 0, 0])
    self.sim_model = self.pybullet_client.loadURDF(
        "laikago/laikago.urdf", 
        self.init_base_pos,
        self.init_base_orn,
        globalScaling=1.0,
        useFixedBase=useFixedBase,
        flags=self.pybullet_client.URDF_MAINTAIN_LINK_ORDER)

    #joint :
    # 0: front-right torso_to_abduct
    # 1: front-right abduct to thigh
    # 2: front-right thigh to knee
    # 3~5: front-left, 6~8: rear-right, 9~11: rear-left
    self.joints = [] #[0,1,2,3,4,5,6,7,8,9,10,11]
    self.force_coef = []
    self.joint_info = []
    self.power = 1.0 #0.2 #1.0
    for j in range(self.pybullet_client.getNumJoints(self.sim_model)): #12개
      joint_info = self.pybullet_client.getJointInfo(self.sim_model, j)
      self.joint_info.append(joint_info)
      if joint_info[2] == 0: #type of joint
        self.joints.append(j)
        self.force_coef.append(joint_info[10])
    self.applied_forces = [0.0 for i in range(len(self.joints))]

    self.init_states = self.get_state()
    self.init_joint_states = self.init_states[13:25]
    for i in range(4):
      #self.init_joint_states[i*3:(i+1)*3] = [0,-0.25,-0.1]
      #self.init_joint_states[i*3:(i+1)*3] = [0,0.12,-1.8]
      self.init_joint_states[i*3:(i+1)*3] = [0,-0.4, 0.27]
      #self.init_joint_states[i*3:(i+1)*3] = [0,0.2, -0.5]

    for j in self.joints:
      self.pybullet_client.setJointMotorControl2(self.sim_model, j, self.pybullet_client.VELOCITY_CONTROL, force=0)
    
    '''
    self.foot_pos = np.array([0,0,-0.18]) #from knee axis
    self.knee_org = np.array([0,0,-0.209]) #from thigh axis
    self.thigh_orgs = np.array([[0,-0.062,0],[0,0.062,0],[0,-0.062,0],[0,0.062,0]]) #from abduct axis
    self.abduct_orgs = np.array([[0.19,-0.049,0.0],[0.19,0.049,0.0],[-0.19,-0.049,0.0],[-0.19,0.049,0.0]]) #from base axis
    self.mass = 3.3 + 4*(0.54 + 0.634 + 0.064 + 0.15)
    '''


  def reset(self):
    self.reset_pose()
    self.applied_forces = [0.0 for i in range(len(self.joints))]
    return self.get_state()


  def get_state(self):
    self.base_pos, self.base_orn = self.pybullet_client.getBasePositionAndOrientation(self.sim_model)
    self.base_vel, self.base_ang_vel = self.pybullet_client.getBaseVelocity(self.sim_model)
    self.joint_states = self.pybullet_client.getJointStates(self.sim_model, self.joints)
    self.joint_pos = [state[0] for state in self.joint_states]
    self.joint_vel = [state[1] for state in self.joint_states]
    self.body_contact, self.contact_feet = self.check_contact()

    #state = list(self.base_pos)+list(self.base_orn)+list(self. base_vel)+list(self.base_ang_vel)+joint_pos+joint_vel+contact_feet+[body_contact]
    #state = list(self.base_orn)+list(self. base_vel)+list(self.base_ang_vel)+joint_pos+joint_vel+contact_feet+[body_contact]
    #state = list(self.base_pos)+list(self.base_orn)+list(self. base_vel)+list(self.base_ang_vel)+self.joint_pos+self.joint_vel+self.contact_feet
    #minitaur에서 쓰는 state:
    #state = list(self.joint_pos)+list(self.joint_vel)+list(self.applied_forces)+list(self.base_orn)
    state = list(self.base_orn)+list(self. base_vel)+list(self.base_ang_vel)+self.joint_pos+self.joint_vel+self.contact_feet
    return np.array(state)


  '''
  def get_force_control(self):
    F = self.find_force()
    tau = self.get_torque_from_force(F)
    return tau


  def find_force(self):
    Kpp = 100.0
    Kdp = 10.0

    p_cd = np.array([0.0, 1.0, 0.0])
    v_cd = np.array([0.0, 0.0, 0.0])
    p_c = np.array(self.base_pos) #base position 과 center of mass 의 위치가 같다고 가정
    v_c = np.array(self.base_vel)
    a_cd = Kpp*(p_cd - p_c) + Kdp*(v_cd - v_c)
    
    F = self.mass * (a_cd + np.array([0.0,9.8,0.0]))
    F = np.array([2*F/8, 2*F/8, 2*F/8, 2*F/8])

    #F = np.array([F, F, F, F])*100
    return F


  def get_torque_from_force(self, F):
    #F = [FR, FL, HR, HL]
    base_pos, base_orn = self.base_pos, self.base_orn
    base_org = np.array(base_pos)
    base_rot = self.pybullet_client.getMatrixFromQuaternion(base_orn)
    base_rot = np.reshape(base_rot, (3,3))

    joint_states = self.joint_states
    joint_angles = [state[0] for state in joint_states]

    torques = []
    for leg in range(4):
      knee_rot = y_rot(-joint_angles[2+leg*3])
      thigh_pos = np.matmul(knee_rot, self.foot_pos) + self.knee_org

      thigh_org = self.thigh_orgs[leg]
      thigh_rot = y_rot(-joint_angles[1+leg*3])
      abduct_pos = np.matmul(thigh_rot, thigh_pos) + thigh_org

      abduct_org = self.abduct_orgs[leg]
      abduct_rot = x_rot(joint_angles[0+leg*3])
      base_pos = np.matmul(abduct_rot, abduct_pos) + abduct_org

      abs_pos = np.matmul(base_rot,base_pos) + base_org
      #test = self.pybullet_client.loadURDF("cube_small.urdf", abs_pos)

      base_force = np.matmul(np.transpose(base_rot),-F[leg])
      #base_force = -F[leg]
      abduct_torque = np.cross(base_pos - abduct_org, base_force)[0]
      abduct_force = np.matmul(np.transpose(abduct_rot),base_force)
      thigh_torque = -np.cross(abduct_pos - thigh_org, abduct_force)[1]
      thigh_force = np.matmul(np.transpose(thigh_rot),abduct_force)
      knee_torque = -np.cross(thigh_pos - self.knee_org, thigh_force)[1]
      torques += [abduct_torque, thigh_torque, knee_torque]
    return torques
  '''


  def reset_pose(self):
    self.pybullet_client.resetBasePositionAndOrientation(self.sim_model, self.init_base_pos, self.init_base_orn)
    self.pybullet_client.resetBaseVelocity(self.sim_model, np.zeros(3), np.zeros(3))
    for i,j in enumerate(self.joints):
      self.pybullet_client.resetJointState(self.sim_model, j, self.init_joint_states[i], 0.0) #velocity = 0.0 으로 setting

  def apply_action(self, action):
    #state = self.get_state()
    #Kp = [20,100,50]*4
    #Kd = [3,5,1]*4
    Kp = [20,50,30]*4
    Kd = [3,3,1]*4
    #exp_w = 0.99#0.9
    for i,j in enumerate(self.joints):
      #force = float(np.clip(action[i], -1, +1)) * self.power * self.force_coef[i]
      p_low, p_high = self.joint_info[i][8:10]
      p_d = ((np.clip(action[i], -1.0, 1.0) + 1)/2) * (p_high - p_low) + p_low
      force = Kp[i]*(p_d - self.joint_pos[i]) - Kd[i]*self.joint_vel[i]

      force = np.clip(force, -self.force_coef[i], self.force_coef[i])
      #force = exp_w*self.applied_forces[i] + (1 - exp_w)*force
      self.applied_forces[i] = force
      '''
      force = 0.0
      if i%3 == 1:
        angle = state[10 + i]
        angle_d = state[10 + i + 12]
        force = 100*(0 - angle) - 0.4*angle_d
        if i == 1:
          print('force :',force)
      '''
      self.pybullet_client.setJointMotorControl2(self.sim_model, j, self.pybullet_client.TORQUE_CONTROL, force=force)


  def check_contact(self):
    pts = self.pybullet_client.getContactPoints(self.sim_model)
    contact_feet = [0 for i in range(4)]
    # 2:FR, 5:FL, 8:RR, 11:RL
    feet_list = [2,5,8,11]
    body_contact = 0 #False
    for p in pts:
      if p[1] == self.sim_model:
        part = p[3]
      elif p[2] == self.sim_model:
        part = p[4]
      if not part in feet_list:
        body_contact = 1 #True
      else:
        contact_feet[(part-2)//3] = 1
    return body_contact, contact_feet


  def get_base_pos(self):
    basePos, baseOrn = self.pybullet_client.getBasePositionAndOrientation(self.sim_model)
    return basePos