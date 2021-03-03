#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import sys
sys.path.append('..')
sys.path.append('.')
import numpy as np
import rospy
from envs.door_open_task_env import DoorOpenTaskEnv
import os
from math import *
from tf.transformations import quaternion_from_euler, euler_from_matrix

class DoorPullAndTraverseTaskEnv(DoorOpenTaskEnv):
    def __init__(self,resolution=(64,64), cam_noise=0.0,use_ft=False):
        super(DoorPullAndTraverseTaskEnv, self).__init__(resolution,cam_noise,use_ft)
        self.stage = 'pull'
        self.open = False
        self.delta = 0
        self.success = False
        self.fail = False

    def _set_init(self):
        self.driver.stop()
        self._reset_mobile_robot(1.5,0.5,0.075,3.14)
        self._wait_door_closed()
        self._random_init_mobile_robot()
        self.stage = 'pull'
        self.open = False
        self.delta = 0
        self.success = False
        self.fail = False

    def _take_action(self, action_idx):
        v0,v1 = 0,0
        if self.stage == 'pull':
            _,v0 = self._door_position()
        elif self.stage == 'traverse':
            v0 = self.pose_sensor.robot().position.x

        action = self.action_space[action_idx]
        self.driver.drive(action[0],action[1])
        rospy.sleep(0.5)

        if self.stage == 'pull':
            _,v1 = self._door_position()
            self.delta = v1-v0
            self.open = self._door_is_open()
            self.fail = self._door_pull_failed()
        elif self.stage == 'traverse':
            v1 = self.pose_sensor.robot().position.x
            self.delta = -(v1-v0)
            self.success = self._robot_is_out()
            self.fail = self._door_traverse_failed()

    def _compute_reward(self):
        # divid to pull and traverse with different rewarding functions
        reward = 0
        if self.stage == 'pull':
            if self.open:
                reward = 100
            elif self.fail:
                reward = -10
            else:
                reward = 10*self.delta - 0.1
        elif self.stage == 'traverse':
            if self.success:
                reward = 100
            elif self.fail:
                reward = -10
            else:
                reward = 10*self.delta - 0.1
        return reward

    def _is_done(self):
        # switch stage
        if self.stage == 'pull' and self.open:
            self.stage = 'traverse'
            print("door is open, robot is traversing the doorway")

        if self.success or self.fail:
            return True
        else:
            return False

    def _random_init_mobile_robot(self):
        #cx = 0.01*(np.random.uniform()-0.5)+0.07 # for door_room
        cx = 0.01*(np.random.uniform()-0.5)+0.02 # for office_room
        cy = 0.01*(np.random.uniform()-0.5)+0.95
        #theta = 0.1*(np.random.uniform()-0.5)+pi # for door_room
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
                return True
        return False

    def _door_traverse_failed(self):
        if not self._robot_is_out():
            campose_r, campose_a = self._camera_position()
            doorpose_r, doorpos_a = self._door_position()
            if campose_r > 1.1*doorpose_r or campose_a > 1.1*doorpos_a:
                return True
        return False
#
