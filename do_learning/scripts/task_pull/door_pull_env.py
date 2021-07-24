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
    def __init__(self,resolution=(64,64),camera='all',cam_noise=0.0, use_force=True, door_width=0.9, door_swinging="left"):
        super(DoorPullEnv, self).__init__(resolution, camera, cam_noise, door_width, door_swinging)
        self.delta = 0 # door angle change by robot action
        self.success = False
        self.fail = False
        self.safe = True
        self.force_in_reward = use_force
        self.radref = None

    # radref = [x,y,theta]
    def set_random_reference(self, radref=None):
        self.radref = radref

    def _set_init(self):
      self.driver.stop()
      if self.door_swinging == "left":
          self._reset_mobile_robot(1.5,0.5,0.075,3.14)
      else:
          self._reset_mobile_robot(-1.5,0.5,0.075,0)

      self._wait_door_closed()
      self._random_init_mobile_robot()
      self.delta = 0
      self.success = False
      self.fail = False
      self.safe = True
      self.tf_sensor.reset_filtered()

    def filtered_force_record(self):
        return self.tf_sensor.filtered()

    def _take_action(self, action_idx):
      self.tf_sensor.reset_step() # get force data during operation

      _,angle0 = self._door_position()
      action = self.action_space[action_idx]
      self.driver.drive(action[0],action[1])
      rospy.sleep(0.5)
      _,angle1 = self._door_position()
      # update
      self.delta = angle1-angle0
      self.success = self._door_is_open()
      self.fail = self._door_pull_failed()

      self.safe = self._safe_contact(self.tf_sensor.step())

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
      return reward

    def _is_done(self):
      if self.success or self.fail:
          return True
      else:
          return False

    def _random_init_mobile_robot(self):
        radref = None
        if self.radref == None:
            radref = np.random.uniform(size=3)
        else:
            radref = self.radref
        cx = 0.01*(radref[0]-0.5)+0.02
        cy = 0.01*(radref[1]-0.5)+self.door_dim[0]+0.05
        theta = 0.5*(radref[2]-0.5)+pi
        ly = -0.25
        if self.door_swinging == "right":
            cx = 0.01*(radref[0]-0.5)-0.065
            cy = 0.01*(radref[1]-0.5)+self.door_dim[0]+0.05
            theta = 0.5*(radref[2]-0.5)
            ly = 0.25

        camera_pose = np.array([[cos(theta),sin(theta),0,cx],
                                [-sin(theta),cos(theta),0,cy],
                                [0,0,1,0.075],
                                [0,0,0,1]])
        mat = np.array([[1,0,0,0.5],
                        [0,1,0,ly],
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
            doorpose_r, doorpose_a = self._door_position()
            print(campose_r, campose_a, doorpose_r, doorpose_a)
            if campose_r > 1.1*doorpose_r or campose_a > 1.1*doorpose_a:
                print("fail to pull the door.")
                return True
        return False
