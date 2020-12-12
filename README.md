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

##
