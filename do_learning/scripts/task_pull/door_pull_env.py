#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import sys
sys.path.append('..')
sys.path.append('.')
import numpy as np
import rospy
from tf.transformations import quaternion_from_euler, euler_from_matrix
from envs.door_open_env import DoorOpenEnv
import os
import tensorflow as tf
from math import *

#
class DoorPullEnv(DoorOpenEnv):
    def __init__(self,resolution=(64,64),cam_noise=0.0, use_force=True):
        super(DoorPullEnv, self).__init__(resolution, cam_noise)
        self.delta = 0 # door angle change by robot action
        self.success = False
        self.fail = False
        self.safe = True
        self.force_in_reward = use_force

    def _set_init(self):
      self.driver.stop()
      self._reset_mobile_robot(1.5,0.5,0.075,3.14)
      self._wait_door_closed()
      self._random_init_mobile_robot()
      self.delta = 0
      self.success = False
      self.fail = False
      self.safe = True

    def _take_action(self, action_idx):
      _,angle0 = self._door_position()
      action = self.action_space[action_idx]
      self.info["action"] = action
      self.driver.drive(action[0],action[1])
      rospy.sleep(0.5)
      _,angle1 = self._door_position()
      # update
      self.delta = angle1-angle0
      self.info["delta_angle"] = self.delta
      self.success = self._door_is_open()
      self.fail = self._door_pull_failed()
      self.safe = self._safe_contact()

    def _compute_reward(self):
      reward = 0
      if self.success:
          reward = 100
      elif self.fail:# or not self.safe:
          reward = -10
      else:
          penalty = 0.1; # step penalty
          if not self.safe and self.force_in_reward:
              penalty += 1 # force safe panalty
          reward = 10*self.delta - penalty
      self.info["reward"] = reward
      return reward

    def _is_done(self):
      if self.success or self.fail:
          return True
      else:
          return False

    def _random_init_mobile_robot(self):
        cx = 0.01*(np.random.uniform()-0.5)+0.02 # for office_room
        cy = 0.01*(np.random.uniform()-0.5)+0.95
        theta = 0.5*(np.random.uniform()-0.5)+pi
        camera_pose = np.array([[cos(theta),sin(theta),0,cx],
                                [-sin(theta),cos(theta),0,cy],
                                [0,0,1,0.075],
                                [0,0,0,1]])
        mat = np.array([[1,0,0,0.5],
                        [0,1,0,-0.25],
                        [0,0,1,0],
                        [0,0,0,1]])
        R = np.dot(camera_pose,np.linalg.inv(mat));
        euler = euler_from_matrix(R[0:3,0:3], 'rxyz')
        robot_x = R[0,3]
        robot_y = R[1,3]
        robot_z = R[2,3]
        yaw = euler[2]
        self._reset_mobile_robot(robot_x,robot_y,robot_z,yaw)

    # check the position of camera
    # if it is in the door block, still trying
    # else failed, reset env
    def _door_pull_failed(self):
        if not self._robot_is_out():
            campose_r, campose_a = self._camera_position()
            doorpose_r, doorpos_a = self._door_position()
            if campose_r > 1.1*doorpose_r or campose_a > 1.1*doorpos_a:
                print("fail to pull the door.")
                return True
        return False
