# door_open_rl
Door opening with a wheeled robot using Reinforcement Learning technique

## start gazebo environment
```
roslaunch do_gazebo door_opening.launch
```

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

## Switch to old environment (door_room.world)
The latest environment (office_room.world) is enriched with, customize door model, lighting and specific ode parameters, while old environment (door_room.world) does not have this features, it use the door from gazebo model base, which contains imperfect mesh and may result bad dynamic simulation performance, but many previous work are done with old environment, so if you want reproduce the old work, you can switch to using old environment.
1. change to use old world file in "/do_gazebo/launch/door_opening.launch"
2. change the friction parameters for wheel to 0.1 in "/do_description/urdf/fourwheeler_base.urdf.xacro"
3. change door dimension to [0.9144 0.0698] and corresponding initial position of robot base in classes DoorOpenTaskEnv and DoorPullTaskEnv
The trained policies with old environment is located in "do_training/scripts/trained_policies/old"
