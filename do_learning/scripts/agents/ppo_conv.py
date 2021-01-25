import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import scipy.signal
import os

"""
CNN-based nerual network for handling images input and output action prediction
'softmax', or normalized exponetial function, is a generalization of the logistic function
to multiple dimensions. It is used in multinomial logistic regression and is often used as
the last activation function of a neural network to normalized the output of a network to a
probability distribution over predicted output classes.
"""
def conv_net(inputs_dim, outputs_dim, outputs_activation='softmax'):
    inputs = keras.Input(shape=inputs_dim, name='images')
    x = layers.Conv2D(32,(3,3), padding='same', activation='relu')(inputs)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(x)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    # outputs
    outputs = layers.Dense(outputs_dim, activation=outputs_activation, name='predictions')(x)
    return keras.Model(inputs=inputs, outputs=outputs)

"""
Replay Buffer, strore experiences and calculate total rewards, advanteges
the buffer will be used for update the policy
"""
class ReplayBuffer:
    def __init__(self, input_shape, action_size, size=1000):
        self.obs_buf = np.zeros([size]+list(input_shape), dtype=np.float32) # states
        self.act_buf = np.zeros((size, action_size), dtype=np.float32) # action, based on stochasitc policy with teh probability
        self.rew_buf = np.zeros(size, dtype=np.float32) # step reward
        self.pred_buf = np.zeros((size, action_size), dtype=np.float32) # prediction: action probability, output of actor net
        self.val_buf = np.zeros(size, dtype=np.float32) # value of (s,a), output of critic net
        self.adv_buf = np.zeros(size, dtype=np.float32) # advantege Q(s,a)-V(s)
        self.ret_buf = np.zeros(size, dtype=np.float32) # ep_return, total reward of episode
        self.ptr, self.idx = 0, 0 # buffer ptr, and current trajectory start index

    def store(self, state, action, reward, prediction, value):
        #print("storing", state[0].shape, action.shape, reward, prediction.shape, value.shape)
        self.obs_buf[self.ptr]=state[0]
        self.act_buf[self.ptr]=action
        self.rew_buf[self.ptr]=reward
        self.pred_buf[self.ptr]=prediction
        self.val_buf[self.ptr]=value
        self.ptr += 1

    def size(self):
        return self.ptr

    """
    For each epidode, calculating the total reward and advanteges with specific
    gamma and lamada
    """
    def ep_update(self, gamma=0.99, lamda=0.95):
        """
        magic from rllab for computing discounted cumulative sums of vectors
        input: vector x: [x0, x1, x2]
        output: [x0+discount*x1+discount^2*x2, x1+discount*x2, x2]
        """
        def discount_cumsum(x,discount):
            return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

        ep_slice = slice(self.idx,self.ptr)
        rews = np.append(self.rew_buf[ep_slice],0)
        vals = np.append(self.val_buf[ep_slice],0)
        # rewards-to-go, which is targets for the value function
        self.ret_buf[ep_slice]=discount_cumsum(rews,gamma)[:-1]
        # General Advantege Estimation, and normolizatiom
        deltas = rews[:-1]+gamma*vals[1:]-vals[:-1]
        advs =discount_cumsum(deltas,gamma*lamda)
        self.adv_buf[ep_slice] = (advs - np.mean(advs))/(np.std(advs)+1e-10)
        self.idx = self.ptr

    def get(self):
        s = slice(0,self.ptr)
        data = dict(states=self.obs_buf[s], actions=self.act_buf[s],
                    returns=self.ret_buf[s], predictions=self.pred_buf[s],
                    advantages=self.adv_buf[s])
        self.ptr, self.idx = 0, 0
        return data

"""
The goal of RL is to find an optimal behavior strategy for the agent to obtain optimal rewards. The policy gradient
methods target at modeling and optimizing the policy directly. The policy loss is defined as
    L = E [log pi (a|s)] * AF
where, 'L' is the policy loss, 'E' is the expected, 'log pi(a|s)' log probability of taking the action at that state
'AF' is the advantage.

PPO is an on-policy algorithm which can be used for environments with either discrete or continous actions spaces.
There are two primary variants of PPO: PPO-penalty which approximately solves a KL-constrained update like TRPO,
but penalizes the KL-divergence in the objective function instead of make it a hard constraint; PPO-clip which does
not have a KL-divergence term in the objective and does not have a constraint at all, instead relies on specialized
clipping in the objective function to remove incentives for the new policy to get far from the old policy
This implementation uses PPO-clip.

references:
[1] https://arxiv.org/pdf/1707.06347.pdf
[2] https://spinningup.openai.com/en/latest/algorithms/ppo.html
"""

"""
Actor net
"""
class Actor_Model:
    def __init__(self, input_shape, action_size, clip_ratio, lr):
        self.clip_ratio = clip_ratio
        self.entropy_loss = 0.001
        self.action_size = action_size
        self.actor = self.build_model(input_shape, action_size, lr)

    def build_model(self, input_shape, action_size, lr):
        model = conv_net(inputs_dim=input_shape, outputs_dim=action_size, outputs_activation="softmax")
        model.compile(loss=self.ppo_loss, optimizer=keras.optimizers.Adam(learning_rate=lr))
        return model

    """
    The key part of the PPO-clip
    policy ratio is define as r = pi(a|s) / pi_old(a|s)
    loss = min(r*AF, clip(r, 1-e, 1+e)*AF), where 'e' is the clip ratio,
    and AF is the advantage function AF(s,a)=Q(s,a)-V(s)
    """
    def ppo_loss(self, y_true, y_pred):
        # y_true: np.hstack([advantages, predictions, actions])
        advs,picks,acts = y_true[:,:1],y_true[:,1:1+self.action_size],y_true[:,1+self.action_size:]
        # print(advs, picks, acts)
        # print(y_pred)
        prob = y_pred*acts
        old_prob = picks*acts
        ratio = prob/(old_prob + 1e-10)
        p1 = ratio*advs
        p2 = K.clip(ratio, 1-self.clip_ratio,1+self.clip_ratio)*advs
        actor_loss = K.mean(K.minimum(p1,p2))
        entropy_loss = self.entropy_loss*K.mean(-(y_pred*K.log(y_pred+1e-10)))
        loss = -(actor_loss+entropy_loss)
        return loss

    def predict(self, state):
        digits = self.actor.predict(state)
        #print("actor prediction", digits)
        return digits

    def fit(self,states,y_true,epochs,batch_size):
        hist = self.actor.fit(states, y_true, epochs=epochs, verbose=0, shuffle=True, batch_size=batch_size)
        return hist.history['loss']

    def save(self, path):
        self.actor.save(path)

    def load(self, path, compile):
        self.actor = keras.models.load_model(path,compile=compile)

"""
Critic net
"""
class Critic_Model:
    def __init__(self, input_shape, lr):
        self.critic = self.build_model(input_shape, lr)

    def build_model(self, input_shape, lr):
        model = conv_net(inputs_dim=input_shape, outputs_dim=1,outputs_activation="linear")
        model.compile(loss="mse",optimizer=keras.optimizers.Adam(learning_rate=lr))
        return model

    def predict(self,state):
        digits = self.critic.predict(state)
        #print("critic prediction", digits)
        return digits

    def fit(self,states,y_true,epochs,batch_size):
        hist = self.critic.fit(states, y_true, epochs=epochs, verbose=0, shuffle=True, batch_size=batch_size)
        return hist.history['loss']

    def save(self, path):
        self.critic.save(path)

    def load(self, path, compile):
        self.critic = keras.models.load_model(path,compile=compile)

"""
A PPO agent class using images as input
"""
class PPOConvAgent:
    def __init__(
        self,
        state_dim,
        action_size,
        clip_ratio=0.2,
        lr_a=1e-4,
        lr_c=3e-4,
    ):
        self.name = 'ppo_conv'
        self.action_size = action_size
        self.Actor = Actor_Model(state_dim,action_size,clip_ratio,lr_a)
        self.Critic = Critic_Model(state_dim,lr_c)

    def action(self, state):
        visual_state = np.expand_dims(state[0], axis=0)# visual state only
        pred = np.squeeze(self.Actor.predict(visual_state), axis=0)
        act = np.random.choice(self.action_size,p=pred) # index of actions
        val = np.squeeze(self.Critic.predict(visual_state), axis=0)
        # print("prediction, action, value:", pred, act, val)
        return pred, act, val

    def train(self, data, batch_size, iter_a=80, iter_c=80):
        states = data['states']
        actions = np.vstack(data['actions'])
        predictions = np.vstack(data['predictions'])
        advantages = np.vstack(data['advantages'])
        returns = np.vstack(data['returns'])
        # stack everything to numpy array
        y_true = np.hstack([advantages, predictions, actions])
        print("ppo training with ",batch_size," pieces of experiences...")
        # training Actor and Crtic networks
        loss_a = self.Actor.fit(states, y_true, iter_a, batch_size)
        print("actor training loss:", loss_a)
        loss_c = self.Critic.fit(states, returns, iter_c, batch_size)
        print("critic training loss:", loss_c)

    def save(self, actor_path, critic_path):
        # save logits_net
        if not os.path.exists(os.path.dirname(actor_path)):
            os.makedirs(os.path.dirname(actor_path))
        self.Actor.save(actor_path)
        # save val_net
        if not os.path.exists(os.path.dirname(critic_path)):
            os.makedirs(os.path.dirname(critic_path))
        self.Critic.save(critic_path)

    def load(self, actor_path, critic_path, compile=False):
        self.Actor.load(actor_path, compile)
        self.Critic.load(critic_path, compile)
