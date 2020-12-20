#!/usr/bin/env python

from __future__ import absolute_import, division, print_function
import sys
sys.path.append('..')
sys.path.append('.')
import numpy as np
import rospy
from tf.transformations import quaternion_from_euler, euler_from_matrix
from envs.door_open_task_env import DoorOpenTaskEnv
from agents.dqn_conv import DQNAgent
from agents.ppo_conv import PPOAgent
import os
import tensorflow as tf
from math import *


class DoorTraverseTaskEnv(DoorOpenTaskEnv):
    def __init__(self,resolution=(64,64),cam_noise=0.0,pull_policy='dqn',pull_model='dqn_noise0.0'):
        super(DoorTraverseTaskEnv, self).__init__(resolution,cam_noise)
        self.door_pull_policy = pull_policy
        self.door_pull_agent = self._load_door_pull_agent(pull_policy,pull_model)
        self.open = False
        self.delta = 0 # robot position change in x direction by robot action
        self.success = False
        self.fail = False

    def _set_init(self):
        self.driver.stop()
        self._reset_mobile_robot(1.5,0.5,0.075,3.14)
        self._wait_door_closed()
        self._reset_mobile_robot(0.61,0.77,0.075,3.3)
        self.open = self._pull_door()
        self.delta = 0 # door angle change by robot action
        self.success = False
        self.fail = False

    def _take_action(self, action_idx):
        x0 = self.pose_sensor.robot().position.x
        action = self.action_space[action_idx]
        self.driver.drive(action[0],action[1])
        rospy.sleep(0.5)
        x1 = self.pose_sensor.robot().position.x
        # update
        self.delta = -(x1-x0)
        self.success = self._robot_is_out()
        self.fail = self._door_traverse_failed()

    def _compute_reward(self):
        reward = 0
        if self.success:
            reward = 100
        elif self.fail:
            reward = -10
        else:
            reward = 10*self.delta - 0.1
        return reward

    def _is_done(self):
        if not self.open or self.success or self.fail:
            return True
        else:
            return False

    def _door_traverse_failed(self):
        if not self._robot_is_out():
            campose_r, campose_a = self._camera_position()
            doorpose_r, doorpos_a = self._door_position()
            if campose_r > 1.1*doorpose_r or campose_a > 1.1*doorpos_a:
                return True
        return False

    def _pull_door(self):
        max_steps = 30
        agent = self.door_pull_agent
        if self.door_pull_policy == 'ppo':
            obs = self._get_observation()
            img = obs.copy()
            for st in range(max_steps):
                act, _, _ = agent.pi_of_a_given_s(np.expand_dims(img, axis=0))
                obs,rew,done,info = self.step(act)
                img = obs.copy()
                if done:
                    break
        else: # dqn
            obs = self._get_observation()
            img = obs.copy()
            for st in range(max_steps):
                act = agent.epsilon_greedy(img)
                obs,rew,done,info = self.step(act)
                img = obs.copy()
                if done:
                    break

        if not self._door_is_open():
            print("door pull failed.")
            return False
        else:
            return True

    def _load_door_pull_agent(self,pull_policy,pull_model):
        if pull_policy == 'ppo':
            agent = PPOAgent(env_type='discrete',dim_obs=(64,64,3),dim_act=self.action_dimension())
            actor_path = os.path.join(sys.path[0], '..', "trained_policies", "door_pull", pull_model, "logits_net")
            critic_path = os.path.join(sys.path[0], '..', "trained_policies", "door_pull", pull_model, "val_net")
            agent.actor.logits_net = tf.keras.models.load_model(actor_path)
            agent.critic.val_net = tf.keras.models.load_model(critic_path)
            return agent
        else:
            agent = DQNAgent(dim_img=(64,64,3),dim_act=self.action_dimension())
            model_path = os.path.join(sys.path[0], '..', 'trained_policies', 'door_pull', pull_model)
            agent.dqn_active = tf.keras.models.load_model(model_path)
            agent.epsilon = 0.0 # determinitic action without random choice
            return agent
