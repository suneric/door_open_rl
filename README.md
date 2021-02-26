# door_open_rl
Door opening with a wheeled robot using Reinforcement Learning techniques

## Prerequisite
- Ubuntu 18.04
- [ROS Melodic](http://wiki.ros.org/melodic/Installation/Ubuntu)
  - Gazebo 9.0, default installed along with ros-melodic-desktop-full or can be installed by following [instruction](http://gazebosim.org/tutorials?tut=install_ubuntu)
    - for [issue of "Error in REST request" when starting gazebo](https://answers.gazebosim.org//question/25030/gazebo-error-restcc205-error-in-rest-request/)
    replace url: https://api.ignitionfuel.org by url: https://api.ignitionrobotics.org inside ~/.ignition/fuel/config.yaml
  - Setup ROS Environment (export below variable in ~/.bashrc)
    - ROS_HOSTNAME=localhost
    - ROS_MASTER_URI=http://localhost:11311
- Install opencv and opencv-python
```
sudo apt install libopencv-dev
python2 pip install opencv-python=4.2.0.32
```
- Install [Realsense SDK](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md) and [Realsense-ros](https://github.com/IntelRealSense/realsense-ros)
```
# add repository first, then install realsense sdk
sudo apt-get install librealsense2-dkms
sudo apt-get install librealsense2-utils
# install realsense-ros
sudo apt-get install ros-melodic-realsense2-camera
```

## dependencies
- tensorflow 2.1.0 (last version support python 2.7)
```
pip install tensorflow==2.1.0
```
- gym
```
pip install gym
```
- scikit-image
```
pip install scikit-image
```

## start gazebo environment
```
roslaunch do_gazebo door_opening.launch
```
- office_room_0.world is for Geforce RTX 1070
- offce_room.world is for Geforce RTX 3070, remove one lighting source


## train specific tasks
- door pulling task
```
cd ../do_training/scripts/task_pull
python door_pull_training_ppo.py
```

- door pushing task
```
cd ../do_training/scripts/task_push
python door_push_training_ppo.py
```

- door traversing task
```
cd ../do_training/scripts/task_traverse
python door_traverse_training_ppo.py
```

## training log
```
tensorboard --logdir=saved_models/task_pull
```

## Switch to old environment (door_room.world)
The latest environment (office_room.world) is enriched with, customize door model, lighting and specific ode parameters, while old environment (door_room.world) does not have this features, it use the door from gazebo model base, which contains imperfect mesh and may result bad dynamic simulation performance, but many previous work are done with old environment, so if you want reproduce the old work, you can switch to using old environment.
1. change to use old world file in "/do_gazebo/launch/door_opening.launch"
2. change the friction parameters for wheel to 0.1 in "/do_description/urdf/fourwheeler_base.urdf.xacro"
3. change door dimension to [0.9144 0.0698] and corresponding initial position of robot base in classes DoorOpenTaskEnv and DoorPullTaskEnv
The trained policies with old environment is located in "do_training/scripts/trained_policies/old"
