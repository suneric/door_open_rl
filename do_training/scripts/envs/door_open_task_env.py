#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import numpy as np
import rospy
from .gym_gazebo_env import GymGazeboEnv
from gym.envs.registration import register
from std_msgs.msg import Float64
from gazebo_msgs.msg import LinkStates, ModelStates, ModelState, LinkState
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Pose, Twist
import tf.transformations as tft
import math
import cv2
from .sensors import CameraSensor, PoseSensor
from .robot_driver import RobotDriver

class ModelSaver:
    def __init__(self,freq):
        self.save_freq = freq

    def save(self,ep,dir,net):
        if not (ep+1)%self.save_freq:
            model_path = os.path.join(dir, str(ep+1))
            if not os.path.exists(os.path.dirname(model_path)):
                os.mkdirs(os.path.dirname(model_path))
            net.save(model_path)
            print("save trained model:", model_path)

register(
  id='DoorOpenTash-v0',
  entry_point='envs.door_open_task_env:DoorOpenTaskEnv')

class DoorOpenTaskEnv(GymGazeboEnv):

  def __init__(self,resolution=(64,64),cam_noise=0.0):
    """
    Initializes a new DoorOpenTaskEnv environment, with define the image size
    and camera noise level (gaussian noise variance, the mean is 0.0)
    """
    super(DoorOpenTaskEnv, self).__init__(
      start_init_physics_parameters=False,
      reset_world_or_sim="WORLD"
    )

    self.info = {}
    self.door_dim = [0.9144, 0.0698] # length, width
    self.action_space = self._action_space()

    rospy.logdebug("Start DoorOpenTaskEnv INIT...")
    self.gazebo.unpauseSim()

    self.front_camera = CameraSensor(resolution,'/cam_front/image_raw',cam_noise)
    self.back_camera = CameraSensor(resolution,'/cam_back/image_raw',cam_noise)
    self.up_camera = CameraSensor(resolution,'/cam_up/image_raw',cam_noise)

    self.pose_sensor = PoseSensor()
    self.driver = RobotDriver()

    self._check_all_sensors_ready()
    self.robot_pose_pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
    self._check_publisher_connection()

    self.gazebo.pauseSim()
    rospy.logdebug("Finished DoorOpenTaskEnv INIT...")

  # a safe velocity of robot is 1.5 m/s, given a radius of 0.5 m robot, the safe angular
  # velocity can be pi rad/s
  def _action_space(self):
      lv,av = 1.5,3.14
      action_space = 2*np.array([[lv,av],[lv,0],[0,av],[-lv,av],[-lv,0],[lv,-av],[0,-av],[-lv,-av]])
      print("action space", action_space)
      return action_space

  def action_dimension(self):
      dim = self.action_space.shape
      print("action space dimension", dim[0])
      return dim[0]

  def _check_all_systems_ready(self):
    """
    Checks that all the sensors, publishers and other simulation systems are
    operational.
    """
    self._check_all_sensors_ready()
    self._check_publisher_connection()

  def _check_all_sensors_ready(self):
    self.front_camera.check_camera_ready()
    self.back_camera.check_camera_ready()
    self.up_camera.check_camera_ready()
    self.pose_sensor.check_sensor_ready()
    rospy.logdebug("All Sensors READY")

  def _check_publisher_connection(self):
    self.driver.check_driver_ready()
    rospy.logdebug("All Publishers READY")

  def _get_observation(self):
    self._display_images()
    img_front = self.front_camera.grey_arr()
    img_back = self.back_camera.grey_arr()
    img_up = self.up_camera.grey_arr()
    # (64x64x3)
    obs = np.concatenate((img_front,img_back,img_up),axis=2)
    # print(obs.shape)
    return obs

  def _display_images(self):
      front = self.front_camera.rgb_image
      back = self.back_camera.rgb_image
      up = self.up_camera.rgb_image
      img = np.concatenate((front,back,up),axis=1)
      cv2.namedWindow("front-back-up")
      img = cv2.resize(img, None, fx=0.5, fy=0.5)
      cv2.imshow('front-back-up',img)
      cv2.waitKey(3)

  # return the robot footprint and door position
  def _post_information(self):
      door_radius, door_angle = self._door_position()
      footprint_lf = self._robot_footprint_position(0.25,0.25)
      footprint_lr = self._robot_footprint_position(-0.25,0.25)
      footprint_rf = self._robot_footprint_position(0.25,-0.25)
      footprint_rr = self._robot_footprint_position(-0.25,-0.25)
      camera_pose = self._robot_footprint_position(0.5,-0.25)
      info = {}
      info['door'] = (door_radius,door_angle)
      info['robot'] = [(footprint_lf[0,3],footprint_lf[1,3]),
                    (footprint_rf[0,3],footprint_rf[1,3]),
                    (footprint_lr[0,3],footprint_lr[1,3]),
                    (footprint_rr[0,3],footprint_rr[1,3]),
                    (camera_pose[0,3], camera_pose[1,3])]
      return info

  #############################################################################
  # overidde functions
  def _set_init(self):
    raise NotImplementedError()

  def _take_action(self, action_idx):
    raise NotImplementedError()

  def _is_done(self):
    raise NotImplementedError()

  def _compute_reward(self):
    raise NotImplementedError()

  #############################################################################
  # utility functions
  def _reset_mobile_robot(self,x,y,z,yaw):
      robot = ModelState()
      robot.model_name = 'mobile_robot'
      robot.pose.position.x = x
      robot.pose.position.y = y
      robot.pose.position.z = z
      rq = tft.quaternion_from_euler(0,0,yaw)
      robot.pose.orientation.x = rq[0]
      robot.pose.orientation.y = rq[1]
      robot.pose.orientation.z = rq[2]
      robot.pose.orientation.w = rq[3]
      self.robot_pose_pub.publish(robot)

  def _wait_door_closed(self):
      door_r, door_a = self._door_position()
      while door_a > 0.11:
          rospy.sleep(0.5)
          door_r, door_a = self._door_position()

  def _door_is_open(self):
      door_r, door_a = self._door_position()
      if door_a > 0.45*math.pi: # 81 degree
          return True
      else:
          return False

  def _robot_is_in(self):
      # footprint of robot
      dr, da = self._door_position()
      fp_lf = self._robot_footprint_position(0.25,0.25)
      fp_lr = self._robot_footprint_position(-0.25,0.25)
      fp_rf = self._robot_footprint_position(0.25,-0.25)
      fp_rr = self._robot_footprint_position(-0.25,-0.25)
      cam_p = self._robot_footprint_position(0.5,-0.25)
      d_x = dr*math.sin(da)
      if fp_lf[0,3] > d_x and fp_lr[0,3] > d_x and fp_rf[0,3] > d_x and fp_rr[0,3] > d_x and cam_p[0,3] > d_x:
          return True
      else:
          return False

  # robot is out of the door way (x < 0)
  def _robot_is_out(self):
     # footprint of robot
     fp_lf = self._robot_footprint_position(0.25,0.25)
     fp_lr = self._robot_footprint_position(-0.25,0.25)
     fp_rf = self._robot_footprint_position(0.25,-0.25)
     fp_rr = self._robot_footprint_position(-0.25,-0.25)
     cam_p = self._robot_footprint_position(0.5,-0.25)
     if fp_lf[0,3] < 0.0 and fp_lr[0,3] < 0.0 and fp_rf[0,3] < 0.0 and fp_rr[0,3] < 0.0 and cam_p[0,3] < 0.0:
         return True
     else:
         return False

  # utility function
  def _robot_footprint_position(self,x,y):
      robot_matrix = self._pose_matrix(self.pose_sensor.robot())
      footprint_trans = np.array([[1,0,0,x],
                               [0,1,0,y],
                               [0,0,1,0],
                               [0,0,0,1]])
      fp_mat = np.dot(robot_matrix, footprint_trans)
      return fp_mat

  # camera position in door polar coordinate frame
  # return radius to (0,0) and angle 0 for (0,1,0)
  def _camera_position(self):
     cam_pose = self._robot_footprint_position(0.5,-0.25)
     angle = math.atan2(cam_pose[0,3],cam_pose[1,3])
     radius = math.sqrt(cam_pose[0,3]*cam_pose[0,3]+cam_pose[1,3]*cam_pose[1,3])
     return radius, angle

  # door position in polar coordinate frame
  # retuen radius to (0,0) and angle 0 for (0,1,0)
  def _door_position(self):
     door_matrix = self._pose_matrix(self.pose_sensor.door())
     door_edge = np.array([[1,0,0,self.door_dim[0]],
                           [0,1,0,0],
                           [0,0,1,0],
                           [0,0,0,1]])
     door_edge_mat = np.dot(door_matrix, door_edge)
     # open angle [0, pi/2]
     open_angle = math.atan2(door_edge_mat[0,3],door_edge_mat[1,3])
     return self.door_dim[0], open_angle

  # convert quaternion based pose to matrix
  def _pose_matrix(self,cp):
      p = cp.position
      q = cp.orientation
      t_mat = tft.translation_matrix([p.x,p.y,p.z])
      r_mat = tft.quaternion_matrix([q.x,q.y,q.z,q.w])
      return np.dot(t_mat,r_mat)
