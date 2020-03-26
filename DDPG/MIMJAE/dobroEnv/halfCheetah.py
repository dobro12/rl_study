from pybullet_utils import bullet_client
import pybullet as p1
import pybullet_data

from copy import deepcopy
import numpy as np
import random
import copy
import time


class HalfCheetah(object):
    def __init__( self, pybullet_client, useFixedBase=True, arg_parser=None):
        self.pybullet_client = pybullet_client

        self.init_base_pos = [0, -0.1, 0]
        self.init_base_orn = self.pybullet_client.getQuaternionFromEuler([-np.pi/2, 0, 0])
        [self.sim_model] = self.pybullet_client.loadMJCF( \
            "mujoco/half_cheetah.xml", \
            flags=p1.URDF_USE_SELF_COLLISION | \
            p1.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS )
        self.pybullet_client.resetBasePositionAndOrientation(self.sim_model, self.init_base_pos, self.init_base_orn)
    
        self.action = []
        self.joints = []
        self.force_coef = []
        self.joint_info = []
        self.base_idx = -1
        for j in range(self.pybullet_client.getNumJoints(self.sim_model)):
            joint_info = self.pybullet_client.getJointInfo(self.sim_model, j)
            self.joint_info.append(joint_info)
            if joint_info[12] == b'torso':
                self.base_idx = j
            if joint_info[2] == 0 and not b'ignore' in joint_info[1]:
                self.joints.append(j)
        self.applied_forces = [0.0 for i in range(len(self.joints))]
        self.force_coef = [1 for a in self.joints]

        #activate torque controller
        for j in self.joints:
            self.pybullet_client.setJointMotorControl2(self.sim_model, j, self.pybullet_client.VELOCITY_CONTROL, force=0)
        
        self.state_id = self.pybullet_client.saveState()

    def reset(self):
        self.action = []
        self.pybullet_client.restoreState(self.state_id)
        self.applied_forces = [0.0 for i in range(len(self.joints))]
        return self.get_state()

    def get_state(self):
        #self.base_pos, self.base_orn = self.pybullet_client.getBasePositionAndOrientation(self.sim_model)
        #self.base_vel, self.base_ang_vel = self.pybullet_client.getBaseVelocity(self.sim_model)
        #self.base_pos, self.base_orn, _, _, _, _ = self.pybullet_client.getLinkState(self.sim_model, self.base_idx)
        self.base_pos, self.base_orn, _, _, _, _, self.base_vel, self.base_ang_vel = \
                self.pybullet_client.getLinkState(self.sim_model, self.base_idx, computeLinkVelocity=1)
        self.base_orn_euler = self.pybullet_client.getEulerFromQuaternion(self.base_orn)
        self.joint_states = self.pybullet_client.getJointStates(self.sim_model, self.joints)
        self.joint_pos = [state[0] for state in self.joint_states]
        self.joint_vel = [state[1] for state in self.joint_states]
        self.body_contact, self.contact_feet = self.check_contact()

        state = list(self.base_pos[0:2])+list(self.base_vel[0:2])+[self.base_orn_euler[2], self.base_ang_vel[2]]\
                +list(self.joint_pos)+list(self.joint_vel)+list(self.contact_feet)
        return np.array(state)

    def check_contact(self):
        body_contact = 0.0
        contact_feet = [0.0, 0.0]
        pts = self.pybullet_client.getContactPoints(self.sim_model)
        for p in pts:
            if p[1] == self.sim_model:
                part = p[3]
            elif p[2] == self.sim_model:
                part = p[4]
            if part == 9:
                contact_feet[0] = 1.0
            elif part == 15:
                contact_feet[1] = 1.0
            else:
                body_contact = 1.0
        return body_contact, contact_feet

    def apply_action(self, action):
        action = np.clip(action, -1.0, 1.0)
        if len(self.action) == 0:
            self.action = action
        else:
            w = 0.5
            self.action = w*self.action + (1-w)*action
            action = self.action
        Kp = 500
        Kd = 6
        self.applied_forces = []
        for i,j in enumerate(self.joints):
            p_low, p_high = self.joint_info[i][8:10]
            #p_d = ((np.clip(action[i], -1.0, 1.0) + 1)/2) * (p_high - p_low) + p_low
            p_d = action[i]
            force = Kp*(p_d - self.joint_pos[i]) - Kd*self.joint_vel[i]
            force = np.clip(force, -100, 100)

            self.applied_forces.append(force)
            self.pybullet_client.setJointMotorControl2(self.sim_model, j, self.pybullet_client.TORQUE_CONTROL, force=force)
