#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import sys
sys.path.append('..')
sys.path.append('.')
import numpy as np
import rospy
from tf.transformations import quaternion_from_euler, euler_from_matrix
from envs.door_open_task_env import DoorOpenTaskEnv
import os
import tensorflow as tf
from math import *

class DoorPushTaskEnv(DoorOpenTaskEnv):
    def __init__(self,resolution=(64,64),cam_noise=0.0):
        super(DoorPushTaskEnv, self).__init__(resolution,cam_noise)
        self.delta = 0 # door angle change by robot action
        self.success = False
        self.fail = False

    def _set_init(self):
        self.driver.stop()
        self._wait_door_closed()
        self._random_init_mobile_robot()
        self.delta = 0 # robot position change in x direction by robot action
        self.success = False
        self.fail = False

    def _take_action(self, action_idx):
        x0 = self.pose_sensor.robot().position.x
        action = self.action_space[action_idx]
        self.driver.drive(action[0],action[1])
        rospy.sleep(0.5)
        x1 = self.pose_sensor.robot().position.x
        # update
        self.delta = x1-x0
        self.success = self._robot_is_in()
        self.fail = self._door_push_failed()

    def _compute_reward(self):
        reward = 0
        if self.success:
            reward = 100
        elif self.fail:
            reward = -10
        else:
            reward = 10*self.delta-0.1
        return reward

    def _is_done(self):
        if self.success or self.fail:
            return True
        else:
            return False

    def _random_init_mobile_robot(self):
        robot_x = 0.1*(np.random.uniform()-0.5)-0.7
        robot_y = 0.1*(np.random.uniform()-0.5)+0.5
        robot_z = 0.075
        yaw = 0.1*pi*(np.random.uniform()-0.5)
        self._reset_mobile_robot(robot_x,robot_y,robot_z,yaw)

    def _door_push_failed(self):
        if self._robot_is_out():
            cam_pose = self._robot_footprint_position(0.5,-0.25)
            cam_x, cam_y = cam_pose[0,3], cam_pose[1,3]
            if cam_x < -1.0 or cam_y < -0.1 or cam_y > 1.1:
                return True
        return False
