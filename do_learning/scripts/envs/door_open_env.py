#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import numpy as np
import rospy
import os
from .gym_gazebo_env import GymGazeboEnv
from gym.envs.registration import register
from std_msgs.msg import Float64
from gazebo_msgs.msg import LinkStates, ModelStates, ModelState, LinkState
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Pose, Twist, WrenchStamped
from sensor_msgs.msg import Image
import tf.transformations as tft
import cv2
from cv_bridge import CvBridge, CvBridgeError
import math
import skimage


"""
CameraSensor with resolution, topic and guassian noise level by default variance = 0.0, mean = 0.0
"""
class CameraSensor():
    def __init__(self, resolution=(64,64), topic='/cam_front/image_raw', noise=0):
        self.resolution = resolution
        self.topic = topic
        self.noise = noise
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(self.topic, Image, self._image_cb)
        self.rgb_image = None
        self.grey_image = None

    def _image_cb(self,data):
        try:
            image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.rgb_image = self._guass_noisy(image, self.noise)
            self.grey_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2GRAY)
        except CvBridgeError as e:
            print(e)

    def check_camera_ready(self):
        self.rgb_image = None
        while self.rgb_image is None and not rospy.is_shutdown():
            try:
                data = rospy.wait_for_message(self.topic, Image, timeout=5.0)
                image = self.bridge.imgmsg_to_cv2(data, "bgr8")
                self.rgb_image = self._guass_noisy(image, self.noise)
                rospy.logdebug("Current image READY=>")
            except:
                rospy.logerr("Current image not ready yet, retrying for getting image")

    def image_arr(self):
        img = cv2.resize(self.rgb_image, self.resolution)
        # normalize the image for easier training
        img_arr = np.array(img)/255 - 0.5
        img_arr = img_arr.reshape((64,64,3))
        return img_arr

    def grey_arr(self):
        img = cv2.resize(self.grey_image, self.resolution)
        # normalize the image for easier training
        img_arr = np.array(img)/255 - 0.5
        img_arr = img_arr.reshape((64,64,1))
        return img_arr

    def _guass_noisy(self,image,var):
        if var > 0:
            img = skimage.util.img_as_float(image)
            noisy = skimage.util.random_noise(img,'gaussian',mean=0.0,var=var)
            return skimage.util.img_as_ubyte(noisy)
        else:
            return image

"""
ForceSensor for the sidebar tip hook joint
"""
class ForceSensor():
    def __init__(self, topic='/tf_sensor_topic'):
        self.topic=topic
        self.force_sub = rospy.Subscriber(self.topic, WrenchStamped, self._force_cb)
        self.record = []
        self.number_of_points = 8
        self.filtered_record = []
        self.step_record = []

    def _force_cb(self,data):
        force = data.wrench.force
        if len(self.record) <= self.number_of_points:
            self.record.append([force.x,force.y,force.z])
        else:
            self.record.pop(0)
            self.record.append([force.x,force.y,force.z])
            self.filtered_record.append(self.data())
            self.step_record.append(self.data())

    def _moving_average(self):
        force_array = np.array(self.record)
        return np.mean(force_array,axis=0)

    # get sensored force data in x,y,z direction
    def data(self):
        return self._moving_average()

    # get force record of entire trajectory
    def reset_filtered(self):
        self.filtered_record = []

    # get force record of a step range
    def reset_step(self):
        self.step_record = []

    def step(self):
        return self.step_record

    def filtered(self):
        return self.filtered_record

    def check_sensor_ready(self):
        self.force_data = None
        while self.force_data is None and not rospy.is_shutdown():
            try:
                data = rospy.wait_for_message(self.topic, WrenchStamped, timeout=5.0)
                self.force_data = data.wrench.force
                rospy.logdebug("Current force sensor READY=>")
            except:
                rospy.logerr("Current force sensor not ready yet, retrying for getting force info")

"""
pose sensor
"""
class PoseSensor():
    def __init__(self, noise=0.0):
        self.noise = noise
        self.door_pose_sub = rospy.Subscriber('/gazebo/link_states', LinkStates, self._door_pose_cb)
        self.robot_pos_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self._robot_pose_cb)
        self.robot_pose = None
        self.door_pose = None

    def _door_pose_cb(self,data):
        index = data.name.index('hinged_door::door')
        self.door_pose = data.pose[index]

    def _robot_pose_cb(self,data):
        index = data.name.index('mobile_robot')
        self.robot_pose = data.pose[index]

    def robot(self):
        return self.robot_pose

    def door(self):
        return self.door_pose

    def check_sensor_ready(self):
        self.robot_pose = None
        rospy.logdebug("Waiting for /gazebo/model_states to be READY...")
        while self.robot_pose is None and not rospy.is_shutdown():
            try:
                data = rospy.wait_for_message("/gazebo/model_states", ModelStates, timeout=5.0)
                index = data.name.index('mobile_robot')
                self.robot_pose = data.pose[index]
                rospy.logdebug("Current  /gazebo/model_states READY=>")
            except:
                rospy.logerr("Current  /gazebo/model_states not ready yet, retrying for getting  /gazebo/model_states")

        self.door_pose = None
        rospy.logdebug("Waiting for /gazebo/link_states to be READY...")
        while self.door_pose is None and not rospy.is_shutdown():
            try:
                data = rospy.wait_for_message("/gazebo/link_states", LinkStates, timeout=5.0)
                index = data.name.index('hinged_door::door')
                self.door_pose = data.pose[index]
                rospy.logdebug("Current  /gazebo/link_states READY=>")
            except:
                rospy.logerr("Current  /gazebo/link_states not ready yet, retrying for getting  /gazebo/link_states")

"""
Robot Driver
"""
class RobotDriver():
    def __init__(self):
        self.vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)

    # give velocities in x direction and z direction with a speed coefficient
    def drive(self,vx,vz,c=1):
        msg = Twist()
        msg.linear.x = vx*c
        msg.linear.y = 0
        msg.linear.z = 0
        msg.angular.x = 0
        msg.angular.y = 0
        msg.angular.z = vz*c
        self.vel_pub.publish(msg)

    def stop(self):
        self.drive(0,0)

    def check_connection(self):
      rate = rospy.Rate(10)  # 10hz
      while self.vel_pub.get_num_connections() == 0 and not rospy.is_shutdown():
        rospy.logdebug("No susbribers to vel_pub yet so we wait and try again")
        try:
          rate.sleep()
        except rospy.ROSInterruptException:
          # This is to avoid error when world is rested, time when backwards.
          pass
      rospy.logdebug("vel_pub Publisher Connected")

###############################################################################
register(
  id='DoorOpen-v0',
  entry_point='envs.door_open_env:DoorOpenEnv')

class DoorOpenEnv(GymGazeboEnv):

    def __init__(self,resolution=(64,64),cam_noise=0.0):
        """
        Initializes a new DoorOpenEnv environment, with define the image size
        and camera noise level (gaussian noise variance, the mean is 0.0)
        """
        super(DoorOpenEnv, self).__init__(
            start_init_physics_parameters=False,
            reset_world_or_sim="WORLD"
        )

        self.door_dim = [0.9, 0.045] # door dimension [length,width]
        self.action_space = self._action_space()

        rospy.logdebug("Start DoorOpenTaskEnv INIT...")
        self.gazebo.unpauseSim()

        self.front_camera = CameraSensor(resolution,'/cam_front/image_raw',cam_noise)
        self.back_camera = CameraSensor(resolution,'/cam_back/image_raw',cam_noise)
        self.up_camera = CameraSensor(resolution,'/cam_up/image_raw',cam_noise)
        self.tf_sensor = ForceSensor('/tf_sensor_topic')
        self.pose_sensor = PoseSensor()
        self._check_all_sensors_ready()

        self.driver = RobotDriver()
        self._check_publisher_connection()
        self.robot_pose_pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)

        self.gazebo.pauseSim()
        rospy.logdebug("Finished DoorOpenTaskEnv INIT...")

    def _action_space(self):
        """
        [Safety Issues in Human-Robot Interactions]
        ISO 10218 states that safe slow speed for a robot needs to be limited to 0.25 m.s-1.
        """
        vx, vz = 1.0, 3.14
        base = np.array([[vx,vz],[vx,0.0],[0.0,vz],[-vx,vz],[-vx,0.0],[vx,-vz],[0.0,-vz],[-vx,-vz]])
        low, high = base,3*base
        action_space = np.concatenate((low,high),axis=0)
        # print(action_space)
        return action_space

    def action_dimension(self):
        dim = self.action_space.shape[0]
        print("action dimension", dim)
        return dim

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
        self.tf_sensor.check_sensor_ready()
        self.pose_sensor.check_sensor_ready()
        rospy.logdebug("All Sensors READY")

    def _check_publisher_connection(self):
        self.driver.check_connection()
        rospy.logdebug("All Publishers READY")

    def _get_observation(self):
        # visual observation (64x64x3)
        img_front = self.front_camera.grey_arr()
        img_back = self.back_camera.grey_arr()
        img_up = self.up_camera.grey_arr()
        images = np.concatenate((img_front,img_back,img_up),axis=2)
        # force sensor information (x,y,z)
        forces = self.tf_sensor.data()
        return (images, forces)

    def _display_images(self):
        front = self.front_camera.rgb_image
        back = self.back_camera.rgb_image
        up = self.up_camera.rgb_image
        img = np.concatenate((front,back,up),axis=1)
        cv2.namedWindow("front-back-up")
        img = cv2.resize(img, None, fx=0.5, fy=0.5)
        cv2.imshow('front-back-up',img)
        cv2.waitKey(1)

    # return the robot footprint and door position
    def _post_information(self):
        self._display_images()
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
    def _safe_contact(self, record, max=70):
        """
        The requirements for door opening force are found in
        the Americans with Disabilities Act Accessibility Guidelines (ADAAG),
        ICC/ANSI A117.1 Standard on Accessible and Usable Buildings and Facilities,
        and the Massachusetts Architectural Access Board requirements (521 CMR)
        - Interior Doors: 5 pounds of force.(22.24111 N)
        - Exterior Doors: 15 pounds of force. (66.72333 N)
        """
        forces = np.array(record)
        max_f = np.max(np.absolute(forces), axis=0)
        print("forces max", max_f)
        danger = any(f > max for f in max_f)
        if danger:
            print("force exceeds safe max: ", max, " N")
            return False
        else:
            return True

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
        # check if reset success
        rospy.sleep(0.2)
        pose = self.pose_sensor.robot()
        if not self._same_position(robot.pose, pose):
            print("required reset to ", robot.pose)
            print("current ", pose)
            self.robot_pose_pub.publish(robot)

    def _same_position(self, pose1, pose2):
        x1, y1 = pose1.position.x, pose1.position.y
        x2, y2 = pose2.position.x, pose2.position.y
        tolerance = 0.001
        if abs(x1-x2) > tolerance or abs(y1-y2) > tolerance:
            return False
        else:
            return True

    def _wait_door_closed(self):
        door_r, door_a = self._door_position()
        while door_a > 0.11:
            rospy.sleep(0.5)
            door_r, door_a = self._door_position()

    def _door_is_open(self):
        door_r, door_a = self._door_position()
        if door_a > 0.45*math.pi: # 81 degree
            print("success to open the door.")
            return True
        else:
            return False

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

    # utility function
    def _robot_footprint_position(self,x,y):
        robot_matrix = self._pose_matrix(self.pose_sensor.robot())
        footprint_trans = np.array([[1,0,0,x],
                                    [0,1,0,y],
                                    [0,0,1,0],
                                    [0,0,0,1]])
        fp_mat = np.dot(robot_matrix, footprint_trans)
        return fp_mat

    # convert quaternion based pose to matrix
    def _pose_matrix(self,cp):
        p = cp.position
        q = cp.orientation
        t_mat = tft.translation_matrix([p.x,p.y,p.z])
        r_mat = tft.quaternion_matrix([q.x,q.y,q.z,q.w])
        return np.dot(t_mat,r_mat)
