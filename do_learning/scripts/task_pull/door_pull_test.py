#!/usr/bin/env python
from __future__ import print_function

import sys
sys.path.append('..')
sys.path.append('.')
import os
import gym
import numpy as np
import time
import random
from gym import wrappers
# ROS packages required
import rospy
import rospkg
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates, LinkStates
import tensorflow as tf
import pickle
import argparse
import matplotlib
import matplotlib.pyplot as plt
import math

from agents.ppo_conv import PPOConvAgent
from agents.ppo_mixed import PPOMixedAgent
from door_pull_env import DoorPullEnv

randref_100 = [[0.09244612, 0.28971416, 0.39807551],
       [0.15705888, 0.86593066, 0.48415634],
       [0.62371232, 0.5695015 , 0.21269476],
       [0.95420665, 0.11396594, 0.57269025],
       [0.08850303, 0.13762764, 0.5965915 ],
       [0.62493148, 0.84724834, 0.47669398],
       [0.50878045, 0.12139389, 0.01357087],
       [0.67287087, 0.70324519, 0.41644886],
       [0.3142239 , 0.21191906, 0.97774347],
       [0.20337789, 0.80662941, 0.94983569],
       [0.64005127, 0.13567923, 0.49286786],
       [0.6557227 , 0.84373298, 0.33607725],
       [0.12174881, 0.67898703, 0.08148927],
       [0.39713955, 0.93397171, 0.9202528 ],
       [0.05236978, 0.72773454, 0.13688932],
       [0.09382604, 0.74765811, 0.74476121],
       [0.81011367, 0.00265018, 0.18989749],
       [0.34265385, 0.35899881, 0.04828679],
       [0.51605478, 0.88347399, 0.84119599],
       [0.96914412, 0.09680524, 0.42177688],
       [0.78340414, 0.91132655, 0.45352022],
       [0.42009788, 0.29367987, 0.65559095],
       [0.7673093 , 0.51520059, 0.55279706],
       [0.79903315, 0.31126498, 0.86263617],
       [0.86549105, 0.48568687, 0.66501608],
       [0.28087118, 0.35590587, 0.3094131 ],
       [0.2704087 , 0.86121824, 0.63265252],
       [0.49614243, 0.20379255, 0.49112456],
       [0.5534246 , 0.00250008, 0.60810921],
       [0.44862653, 0.41796539, 0.22071121],
       [0.98117689, 0.72648968, 0.46219923],
       [0.53021925, 0.98692169, 0.56347823],
       [0.40750443, 0.09082105, 0.00765231],
       [0.96146458, 0.86425629, 0.57591452],
       [0.79937527, 0.73581078, 0.83827976],
       [0.27981576, 0.08513959, 0.77988103],
       [0.03323906, 0.60407472, 0.51870571],
       [0.28869575, 0.56003708, 0.80524858],
       [0.016879  , 0.07560814, 0.67360536],
       [0.00470292, 0.14969729, 0.51369731],
       [0.15255814, 0.79041548, 0.16292368],
       [0.17731367, 0.20719869, 0.50328528],
       [0.50986287, 0.16172861, 0.94573424],
       [0.95133003, 0.17275329, 0.9109839 ],
       [0.28702396, 0.23552682, 0.65070134],
       [0.54025518, 0.6707053 , 0.95857403],
       [0.42519431, 0.25204589, 0.24549153],
       [0.51531292, 0.39486664, 0.12223444],
       [0.82559804, 0.88089863, 0.27075463],
       [0.61413655, 0.94444622, 0.82082929],
       [0.58612847, 0.60765052, 0.76900651],
       [0.85360777, 0.90831105, 0.74866033],
       [0.76628631, 0.08402217, 0.76783893],
       [0.20459046, 0.46826997, 0.90775457],
       [0.514957  , 0.70451124, 0.77385886],
       [0.7919448 , 0.7613035 , 0.24221695],
       [0.20789733, 0.56694471, 0.6940616 ],
       [0.12942064, 0.02579941, 0.78059334],
       [0.54489011, 0.4094482 , 0.12751735],
       [0.00580972, 0.07967641, 0.77947684],
       [0.65708265, 0.27699915, 0.95685873],
       [0.92609372, 0.85096699, 0.63464718],
       [0.22876261, 0.69712388, 0.28030192],
       [0.53113457, 0.41787779, 0.47692947],
       [0.44715777, 0.72698775, 0.52270818],
       [0.28216714, 0.22499236, 0.00530878],
       [0.19416741, 0.69347557, 0.95422665],
       [0.09024857, 0.5237406 , 0.64089686],
       [0.07437527, 0.90856046, 0.46953091],
       [0.7270987 , 0.39988446, 0.07662119],
       [0.13199407, 0.6205841 , 0.25393838],
       [0.93555684, 0.20342696, 0.41933268],
       [0.85142741, 0.27607342, 0.85084012],
       [0.89950758, 0.94035575, 0.06037877],
       [0.70346677, 0.56350346, 0.35283383],
       [0.9194111 , 0.32437353, 0.92770994],
       [0.50435946, 0.2218051 , 0.65720154],
       [0.92098942, 0.45246469, 0.44577439],
       [0.49054438, 0.67860086, 0.3404282 ],
       [0.81212874, 0.18784854, 0.86630749],
       [0.41511007, 0.4635757 , 0.76694083],
       [0.63261966, 0.19907404, 0.55492605],
       [0.82498276, 0.42437836, 0.88419515],
       [0.36639202, 0.39198937, 0.7190022 ],
       [0.30070376, 0.45067663, 0.69478707],
       [0.56186253, 0.41419547, 0.30716667],
       [0.55837551, 0.09002388, 0.17522118],
       [0.58362862, 0.55337339, 0.96835221],
       [0.08152436, 0.4895248 , 0.78676868],
       [0.99124588, 0.38755241, 0.92597022],
       [0.50904568, 0.58452882, 0.38321651],
       [0.14053537, 0.41233305, 0.54420963],
       [0.82348906, 0.49565012, 0.1445384 ],
       [0.0246051 , 0.17306495, 0.84447899],
       [0.75783633, 0.24025567, 0.53463428],
       [0.12553018, 0.91500178, 0.03450488],
       [0.90897501, 0.99887673, 0.69299442],
       [0.45346963, 0.89668278, 0.14203193],
       [0.98730827, 0.92550436, 0.10753562],
       [0.27999242, 0.42432665, 0.74311562]]


def plot_forces(forces, steps, ax):
    force_arr = np.array(forces)
    x = np.linspace(1,steps,force_arr.shape[0])
    ax.plot(x,force_arr[:,0],label="detected force x")
    ax.plot(x,force_arr[:,1],label="detected force y")
    ax.plot(x,force_arr[:,2],label="detected force z")
    ax.set_xlabel('step', fontsize=15)
    ax.set_ylabel('forces (N)', fontsize=15)
    ax.set_xticks(range(1,steps+1))
    ax.legend()

def plot_actions(actions, steps, ax):
    action_arr = np.array(actions)
    x = np.linspace(1,steps,steps)
    ax.plot(x,action_arr,label="action sequence",marker='*',linestyle='dashed',color='k')
    ax.set_xlabel('step', fontsize=15)
    ax.set_ylabel('action', fontsize=15)
    ax.set_xticks(range(1,steps+1))
    ax.set_yticks(range(16))
    ax.set_yticklabels(['L_FL','L_F','L_L','L_BL','L_B','L_FR','L_R','L_BR','H_FL','H_F','H_L','H_BL','H_B','H_FR','H_R','H_BR'])
    ax.legend()

def plot_values(values, steps, ax):
    value_arr = np.array(values)
    x = np.linspace(1,steps,steps)
    ax.plot(x,value_arr,label="action value",marker='o',color='k')
    ax.set_xticks(range(1,steps+1))
    ax.set_xlabel('step', fontsize=15)
    ax.set_ylabel('value', fontsize=15)
    ax.legend()

# plot trajectory which contains a sequence of pose of robot and door
# info {
#       'door': (radius, angle),
#       'robot': [(fp_lf_x, fp_lf_y),
#                 (fp_rf_x, fp_rf_y),
#                 (fp_lr_x, fp_lr_y),
#                 (fp_rr_x, fp_rr_y),
#                 (cam_p_x, cam_p_y)]
#      }
def plot_trajectorty(t, ax):
    # room
    font = {'size':12}
    matplotlib.rc('font',**font)
    ax.set_xlim(-1.5,2.5)
    ax.set_ylim(-3,1)
    ax.axis('equal')
    rx = [0,0,2,2,0,0]
    ry = [0,-0.5,-0.5,1.5,1.5,0.9]
    ax.plot(rx,ry,linewidth=10.0,color='lightgrey')

    cost1, cost2 = trajectory_cost(t)
    txt_info = "Steps:\n"+str(len(t))+"\nRobot Path:\n"+str(round(cost2,3))+"m\nSidebar Path:\n"+str(round(cost1,3))+"m"
    ax.text(-1,-0.5,txt_info,fontsize=12)
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Y (m)", fontsize=12)

    #plt.grid(True)
    door_t = []
    robot_t = []
    cam_t = []
    index = 0
    for info in t:
        dr = info['door'][0]
        da = info['door'][1]
        door_t.append((dr*math.sin(da),dr*math.cos(da)))
        r1 = info['robot'][1]
        r2 = info['robot'][0]
        r3 = info['robot'][2]
        r4 = info['robot'][3]
        r5 = r1
        r6 = info['robot'][4]
        rx = np.mean([r1[0],r2[0],r3[0],r4[0]])
        ry = np.mean([r1[1],r2[1],r3[1],r4[1]])
        robot_t.append([rx,ry])
        cam_t.append(r6)
        # draw first and last footprint
        ax.set_yticks([-1,0,1,2])
        ax.set_xticks([-1,0,1,2,3])
        if index == len(t)-1:
            line1, = ax.plot([0,dr*math.sin(da)],[0,dr*math.cos(da)],linewidth=5,color='y')
            line1.set_label('door')
            line2, = ax.plot([r1[0],r2[0],r3[0],r4[0],r5[0]],[r1[1],r2[1],r3[1],r4[1],r5[1]],linewidth=3,color='dimgrey')
            line2.set_label('robot')
            line3, = ax.plot([r5[0],r6[0]],[r5[1],r6[1]],linewidth=3,color='red')
            line3.set_label('sidebar')
            # local coordinate
            dx = [0.5*(np.mean([r1[0],r2[0]])-rx), 0.5*(np.mean([r1[1],r2[1]])-ry)]
            line4,= ax.plot([rx,rx+dx[0]],[ry,ry+dx[1]],linewidth=1,color='r')
            ax.text(rx+1.2*dx[0],ry+1.2*dx[1],'x',fontsize=8)
            dy = [0.5*(np.mean([r3[0],r2[0]])-rx), 0.5*(np.mean([r3[1],r2[1]])-ry)]
            line5,= ax.plot([rx,rx+dy[0]],[ry,ry+dy[1]],linewidth=1,color='g')
            ax.text(rx+1.2*dy[0],ry+1.2*dy[1],'y',fontsize=8)

            # ax.legend((line1,line2,line3),('door','mobile robot','sidebar'))
        elif index == 0 or index == int(len(t)/2) or index == int(len(t)/4) or index == int(len(t)*3/4):
            ax.plot([0,dr*math.sin(da)],[0,dr*math.cos(da)],linewidth=5,color='y',alpha=0.35)
            ax.plot([r1[0],r2[0],r3[0],r4[0],r5[0]],[r1[1],r2[1],r3[1],r4[1],r5[1]],linewidth=3,color='dimgrey',alpha=0.35)
            ax.plot([r5[0],r6[0]],[r5[1],r6[1]],linewidth=3,color='red',alpha=0.35)
        index += 1
    #ax.plot(np.matrix(door_t)[:,0],np.matrix(door_t)[:,1], linewidth=1.0, color='y')
    line4, = ax.plot(np.matrix(robot_t)[:,0],np.matrix(robot_t)[:,1], linewidth=1.0, color='dimgrey', linestyle='dashed')
    line4.set_label('robot path')
    line5, = ax.plot(np.matrix(cam_t)[:,0],np.matrix(cam_t)[:,1], linewidth=1.0, color='red',linestyle='dashed')
    line5.set_label('sidebar path')
    ax.legend(loc='upper right')

# a simple cost evaluation based on the end of the bar where camera placed at
# total displacement of the end of the bar
def trajectory_cost(t):
    dist1, dist2 = 0,0
    pos1 = t[0]['robot'][4] # side bar pos
    robot = t[0]['robot']
    rx = np.mean([robot[1][0],robot[0][0],robot[2][0],robot[3][0]])
    ry = np.mean([robot[1][1],robot[0][1],robot[2][1],robot[3][1]])
    pos2 = [rx,ry] # robot center pos
    for info in t:
        cam = info['robot'][4]
        displacement1 = math.sqrt((cam[0]-pos1[0])*(cam[0]-pos1[0])+(cam[1]-pos1[1])*(cam[1]-pos1[1]))
        dist1 += displacement1
        pos1 = cam

        r1 = info['robot'][1]
        r2 = info['robot'][0]
        r3 = info['robot'][2]
        r4 = info['robot'][3]
        rc = [np.mean([r1[0],r2[0],r3[0],r4[0]]),np.mean([r1[1],r2[1],r3[1],r4[1]])]
        displacement2 = math.sqrt((rc[0]-pos2[0])*(rc[0]-pos2[0])+(rc[1]-pos2[1])*(rc[1]-pos2[1]))
        dist2 += displacement2
        pos2 = rc

    return dist1, dist2

###############################################################################
# PPO TEST
def run_ppo_test(episode,env,agent,max_steps,radref):
    success_counter = 0
    trajectory_list,values_list,forces_list,actions_list = [],[],[],[]
    for ep in range(episode):
        trajectory,forces,actions,values = [],[],[],[]
        # initial robot position
        env.set_random_reference(radref[ep][:])
        obs, info = env.reset()
        done = False
        for st in range(max_steps):
            pred, act, val = agent.action(obs)
            n_obs, rew, done, info = env.step(act)
            trajectory.append(info)
            actions.append(act)
            values.append(val[0])
            obs = n_obs
            if done:
                forces = env.filtered_force_record()
                break

        if env.success:
            success_counter += 1
            trajectory_list.append(trajectory)
            forces_list.append(forces)
            actions_list.append(actions)
            values_list.append(values)

        print("Succeeded: {} / {}".format(success_counter,ep+1))

    return trajectory_list, forces_list, actions_list, values_list

###############################################################################
# main loop
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--door_width', type=float, default=0.9) # door width
    parser.add_argument('--policy', type=str, default="conv") # dqn, ppo
    parser.add_argument('--eps', type=int, default=10) # test episode
    parser.add_argument('--noise', type=float, default=0.0) # noise variance 0.02
    parser.add_argument('--max_steps', type=int ,default=60)
    parser.add_argument('--camera', type=str, default='all')
    parser.add_argument('--actor_model',type=str, default="ppo_noise0.0/logits_net/10000") # ppo model
    parser.add_argument('--critic_model',type=str, default="ppo_noise0.0/val_net/10000") # ppo model
    parser.add_argument('--use_force_in_reward', default=True)
    parser.add_argument('--plot', default=True)
    return parser.parse_args()

np.random.seed(7)

# the trained models are located in "trained_policies" folder
# ppo pull training:
# python test/door_open_test.py
# --policy=ppo --task=pull --actor_model=ppo_noise0.0/logits_net --critic_model=ppo_noise0.0/val_net
# or specifying iteration with a folder name like ppo_noise0.0/val_net/150

if __name__ == "__main__":
    args = get_args()
    rospy.init_node('door_pull_test', anonymous=True, log_level=rospy.INFO)

    # run specific task with specific policy
    env = DoorPullEnv(resolution=(64,64),camera= args.camera, cam_noise=args.noise, use_force=args.use_force_in_reward, door_width=args.door_width)
    act_dim = env.action_dimension()
    visual_dim = env.visual_dimension()
    agent = PPOConvAgent(state_dim=visual_dim, action_size=act_dim)
    if args.policy == "mixed":
        agent = PPOMixedAgent(image_dim=visual_dim,force_dim=3, action_size=act_dim)
    actor_path = os.path.join(sys.path[0], '..', "trained_policies", "door_pull", args.actor_model)
    critic_path = os.path.join(sys.path[0], '..', "trained_policies", "door_pull", args.critic_model)
    agent.load(actor_path, critic_path)
    trajectories, forces, actions, values = run_ppo_test(args.eps,env,agent,args.max_steps,randref_100[0:args.eps][:])
    if len(trajectories) == 0:
        print("No successful test");
    else:
        # trajectory analysis
        trajectory_steps = [len(i)-1 for i in trajectories]
        print("====================")
        print("Success Rate", len(trajectory_steps),"/", args.eps)

        average_steps = int(sum(trajectory_steps)/len(trajectory_steps))
        print("Average Steps", average_steps)
        print("Minimum Steps", min(trajectory_steps))
        print("Maximum Steps", max(trajectory_steps))

        average_values = [sum(vals)/len(vals) for vals in values]
        print("Average Value", sum(average_values)/len(average_values))
        print("Lowest Value",  min(average_values))
        print("Highest Value", max(average_values))

        max_forces = [np.max(np.absolute(np.array(record)), axis=0) for record in forces]
        max_forces = [np.max(i) for i in max_forces]
        print("Average Max_Force", sum(max_forces)/len(max_forces))
        print("Lowest Max_Force", min(max_forces))
        print("Highet Max_Force", max(max_forces))
        print("====================")

        print("plot trajectory", args.plot)
        if args.plot:
            # plot trajectory with least number of steps
            ls_idx = trajectory_steps.index(min(trajectory_steps))
            steps = len(trajectories[ls_idx])
            fig0,ax0 = plt.subplots(2,2)
            plot_trajectorty(trajectories[ls_idx], ax0[0][0])
            plot_forces(forces[ls_idx], steps, ax0[0][1])
            plot_values(values[ls_idx],steps, ax0[1][0])
            plot_actions(actions[ls_idx], steps, ax0[1][1])
            fig0.suptitle("Case: Least Number of Steps")
            # plot smallest max force example
            smf_idx = max_forces.index(min(max_forces))
            steps = len(trajectories[smf_idx])
            fig1,ax1 = plt.subplots(2,2)
            plot_trajectorty(trajectories[smf_idx], ax1[0][0])
            plot_forces(forces[smf_idx], steps, ax1[0][1])
            plot_values(values[smf_idx],steps, ax1[1][0])
            plot_actions(actions[smf_idx], steps, ax1[1][1])
            fig1.suptitle("Case: Smallest Detected Max Force")

            # plot largest max force example
            lmf_idx = max_forces.index(max(max_forces))
            steps = len(trajectories[lmf_idx])
            fig2,ax2 = plt.subplots(2,2)
            plot_trajectorty(trajectories[lmf_idx], ax2[0][0])
            plot_forces(forces[lmf_idx], steps, ax2[0][1])
            plot_values(values[lmf_idx],steps, ax2[1][0])
            plot_actions(actions[lmf_idx], steps, ax2[1][1])
            fig2.suptitle("Case: Largest Detected Max Force")
            plt.show()
