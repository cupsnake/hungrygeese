from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, Configuration, Action, row_col, translate
import numpy as np
import torch.nn.init as init
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import os
from collections import deque
import pickle
import bz2
import base64
from torch.distributions import Categorical
from statistics import mean

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# settings
Train_max_step         = 4000000
learning_rate          = 1e-4
gamma                  = 0.99
lambd                  = 0.95
eps_clip               = 0.12
K_epoch                =  1
N_worker               = 40
T_horizon              = 20
save_interval          = 300


def getValidMoves(obs, last_obs, index):  

    try: 
        rows=7
        columns=11
        actions=[Action.EAST, Action.SOUTH, Action.WEST, Action.NORTH]
        vactionlist=[]
        # print(len(obs))
        # sdfS
        for i in range(len(obs)):
            geese = obs[i]['geese']
            pos = geese[index][0]
            obstacles1 = {position for goose in geese for position in goose[:-1]}
            print(last_obs)
            if last_obs[i] is not None: 
                obstacles1.add(last_obs[i]['geese'][index][0])
                

            valid_moves1 = [
                translate(pos, action, columns, rows) not in obstacles1
                for action in actions
            ]
            vactionlist.append(valid_moves1)
        return vactionlist
    except:
        print("chjelere")
        print(obs[i])
        print(last_obs)
        
def generateaction(probs,vactionlist):
    oaction=[]
    for i in range(len(vactionlist)):
        revisep = probs[i,:] * vactionlist[i]  # masking invalid moves
        sum_Ps_s = np.sum(revisep)
        if sum_Ps_s > 0:
            revisep /= sum_Ps_s
        else:
            revisep=[0.25,0.25,0.25,0.25]

        action=np.random.choice(4, p=revisep)
        oaction.append(action)
    return oaction

def location(loc):
    row = 7
    column = 11
#     print(loc % column)
#     print(loc // column)
    return loc // column, loc % column

 

# class GeeseNet(nn.Module):
#     def __init__(self):
#         super(GeeseNet, self).__init__()
#         self.cnn1=nn.Conv2d(17, 17, kernel_size=3, stride=1,padding=1)
#         self.cnn2=nn.Conv2d(17, 12, kernel_size=1, stride=1)
#         self.linear2 = nn.Linear(12*7*11, 512)
#         self.linear3 = nn.Linear(512, 256)
#         self.linear4 = nn.Linear(256, 4)
#         self.linear5 = nn.Linear(256, 1)
#     def forward(self, state):
#         feature=F.relu(self.cnn1(state))
#         feature=F.relu(self.cnn2(feature))
#         # x1=torch.flatten(feature) 
#         x1=feature.view(-1,12*7*11) 
#         x1 = F.relu(self.linear2(x1))
#         adv = F.relu(self.linear3(x1))
#         # print(adv.shape)
#         advantages = torch.softmax(self.linear4(adv),1)
#         value = self.linear5(adv)
#         return advantages,value
class TorusConv2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, bn):
        super().__init__()
        self.edge_size = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size)
        self.bn = nn.BatchNorm2d(output_dim) if bn else None

    def forward(self, x):
        h = torch.cat([x[:,:,:,-self.edge_size[1]:], x, x[:,:,:,:self.edge_size[1]]], dim=3)
        h = torch.cat([h[:,:,-self.edge_size[0]:], h, h[:,:,:self.edge_size[0]]], dim=2)
        h = self.conv(h)
        h = self.bn(h) if self.bn is not None else h
        return h

class GeeseNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers, filters = 12, 32
        self.conv0 = TorusConv2d(17, filters, (3, 3), True)
        self.blocks = nn.ModuleList([TorusConv2d(filters, filters, (3, 3), True) for _ in range(layers)])
        self.head_p = nn.Linear(filters, 4, bias=False)
        self.head_v = nn.Linear(filters * 2, 1, bias=False)

    def forward(self, x):
        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = F.relu_(h + block(h))
        h_head = (h * x[:,:1]).view(h.size(0), h.size(1), -1).sum(-1)
        h_avg = h.view(h.size(0), h.size(1), -1).mean(-1)
        p = torch.softmax(self.head_p(h_head), 1)
        v = torch.tanh(self.head_v(torch.cat([h_head, h_avg], 1)))
        return p, v

def train(Net, optimizer, states, actions, rewards, next_states, dones, old_probs):

    states = torch.FloatTensor(states).view(-1, 17,7,11).to(device) 
    actions = torch.LongTensor(actions).view(-1, 1).to(device) 
    rewards = torch.FloatTensor(rewards).view(-1, 1).to(device) 
    next_states = torch.FloatTensor(next_states).view(-1, 17,7,11).to(device) 
    dones = torch.FloatTensor(dones).view(-1, 1).to(device) 
    old_probs = torch.FloatTensor(old_probs).view(-1, 1).to(device) 

    for _ in range(K_epoch):
        probs, values = Net(states) 
        _, next_values = Net(next_states) 

        td_targets = rewards + gamma * next_values * dones 
        deltas = td_targets - values # (T*N, 1)

        # calculate GAE
        deltas = deltas.view(T_horizon, N_worker, 1).cpu().detach().numpy() 
        masks = dones.view(T_horizon, N_worker, 1).cpu().numpy()
        advantages = []
        advantage = 0
        for delta, mask in zip(deltas[::-1], masks[::-1]):
            advantage = gamma * lambd * advantage * mask + delta
            advantages.append(advantage)
        advantages.reverse()
        advantages = torch.FloatTensor(advantages).view(-1, 1).to(device) 

        probs_a = probs.gather(1, actions) #(T*N, 1)

        m = Categorical(probs)
        entropy = m.entropy()

        ratio = torch.exp(torch.log(probs_a) - torch.log(old_probs))
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantages

        actor_loss = -torch.mean(torch.min(surr1, surr2))
        critic_loss = F.smooth_l1_loss(values, td_targets.detach())
        entropy_loss = torch.mean(entropy)

        loss = actor_loss + critic_loss -  0.04*entropy_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

class CustomHungrygeese_stack():
    def __init__(self, env):
        super(CustomHungrygeese_stack, self).__init__()
        self.env=env
        self.paction=None
        self.player_index=3
    def full_custom_reward(self,obs,oldobs,player_index):
        enemylist=[0,1,2,3]
        enemylist.remove(player_index)
        # print(oldobs['geese'][player_index])
        reward=0.
        if len(obs["geese"][enemylist[0]])==0 and len(obs["geese"][enemylist[1]])==0 and len(obs["geese"][enemylist[2]])==0:
                reward=1
                # print("winner ---------------111---------------"+obs)
        maxmem=0
        maxindex=0
        if obs['step']==199:
            for i in range(4):
                if len(obs["geese"][i])>=maxmem:
                    maxmem=len(obs["geese"][i])
                    maxindex=i
            if maxindex==player_index:
                    reward=1
                    # print("winner ---------------2211---------------"+obs)
        if len(obs['geese'][player_index])>len(oldobs['geese'][player_index]):
            reward=0.2
        if len(obs["geese"][player_index])==0:
            reward=-1.5
    #         print("alllllllllllllllert-------")
        
        return reward
    def parseaction(self,rawaction):
        if rawaction==0:
            return Action.NORTH.name
        elif rawaction==1:
            return Action.SOUTH.name
        elif rawaction==2:
            return Action.WEST.name
        elif rawaction==3:
            return Action.EAST.name
            
    def parse_dict1(self,obs,last_obs,index):
        b = np.zeros((17, 7 * 11), dtype=np.float32)
        for p, pos_list in enumerate(obs['geese']):
            # head position
            for pos in pos_list[:1]:
                b[0 + (p - index) % 4, pos] = 1
            # tip position
            for pos in pos_list[-1:]:
                b[4 + (p - index) % 4, pos] = 1
            # whole position
            for pos in pos_list:
                b[8 + (p - index) % 4, pos] = 1
        # previous head position
        if last_obs is not None:
            for p, pos_list in enumerate(last_obs['geese']):
                for pos in pos_list[:1]:
                    b[12 + (p - index) % 4, pos] = 1
        # food
        for pos in obs['food']:
            b[16, pos] = 1

        return b.reshape(-1, 7, 11)

    def step(self, action):
        old_obs=self.old_obs_dict
        obs, reward, done, info = self.env.step(self.parseaction(action))
        reward=self.full_custom_reward(obs,self.obs_dict,self.player_index)
        self.obs_dict=obs
        next_real_image=self.parse_dict1(self.obs_dict,self.old_obs_dict,self.player_index)
        self.old_obs_dict=obs
        # next_real_image=np.concatenate([next_obs_image,self.obs_image])
        next_state= next_real_image
        # self.obs_image=next_obs_image
        return next_state,reward, done, info, obs,old_obs

    def reset(self):
        self.obs_dict = self.env.reset()
        self.paction=None
        # print(self.obs_dict[0])
        self.old_obs_dict=self.obs_dict
        real_image=self.parse_dict1(self.obs_dict,self.obs_dict,self.player_index)
        # real_image=np.concatenate([self.obs_image,self.obs_image])

        state= real_image
        # self.vaction=self.checkstep(self.obs_image,self.paction)
        return state,self.obs_dict

def CreateHungrygeese():
    env = make("hungry_geese") # , debug=True)
    # trainer = env.train(["main.py", "main.py", "main.py", None])
    trainer = env.train(["agent.py", "main.py", "main.py", None])
    env = CustomHungrygeese_stack(trainer)
    return env

class MultipleHungrygeese:
    def __init__(self, N):
        self.envs = [CreateHungrygeese() for _ in range(N)]
    
    def reset(self):
        obs = []
        oobs=[]
        for env in self.envs:
            ob,oob= env.reset()
            obs.append(ob)
            oobs.append(oob)

        return np.stack(obs),np.stack(oobs)
    
    def step(self, actions):
        obs, rewards, dones, infos,orignal_obs,last_obs = [], [], [], [] ,[],[]
        for env, action in zip(self.envs, actions):
            ob, reward, done, info,oobs,loobs = env.step(action)
            # if done:
            #     ob,oobs = env.reset()
            obs.append(ob)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
            orignal_obs.append(oobs)
            last_obs.append(loobs)
        return np.stack(obs), np.stack(rewards), np.stack(dones),np.stack(orignal_obs),np.stack(last_obs)
    def checkdone(self,dones,state,last_obs,nowobs):
        for i in range(len(dones)):
            # print(dones[i])
            if dones[i]:
                print(nowobs[i])
                obs,oobs = self.envs[i].reset()
                
                nowobs[i]=oobs
                state[i]=obs
                last_obs[i]=None
        return state,nowobs,last_obs

def parse_dict1(obs,last_obs,index):
        b = np.zeros((17, 7 * 11), dtype=np.float32)
        for p, pos_list in enumerate(obs['geese']):
            # head position
            for pos in pos_list[:1]:
                b[0 + (p - index) % 4, pos] = 1
            # tip position
            for pos in pos_list[-1:]:
                b[4 + (p - index) % 4, pos] = 1
            # whole position
            for pos in pos_list:
                b[8 + (p - index) % 4, pos] = 1
        # previous head position
        if last_obs is not None:
            for p, pos_list in enumerate(last_obs['geese']):
                for pos in pos_list[:1]:
                    b[12 + (p - index) % 4, pos] = 1

        # food
        for pos in obs['food']:
            b[16, pos] = 1

        return b.reshape(-1, 7, 11)

def main():
    env = MultipleHungrygeese(N_worker)
    Net = GeeseNet().to(device)
    Net.load_state_dict(torch.load('critic8.model',map_location=device))
    optimizer = torch.optim.Adam(Net.parameters(), learning_rate)

    scores = [0.0 for _ in range(N_worker)]
    score_history = []
    train_history = []
    #train_history = np.load(history_path+'.npy').tolist()
    
    step = 0

    state,obs = env.reset() # (N, 4, 84, 84)
    last_obs=[None]*N_worker
    # print(last_obs)
    # sf

    print("Train Start")
    while step <= Train_max_step:
        states, actions, rewards, next_states, dones, old_probs = list(), list(), list(), list(), list(), list()
        for _ in range(T_horizon):
            prob, _ = Net(torch.FloatTensor(state).to(device))
            m = Categorical(prob)

            action = m.sample() 
            old_prob = prob.gather(1, action.unsqueeze(1))

            action = action.cpu().detach().numpy()
            # print(old_prob)
            old_prob = old_prob.cpu().detach().numpy()

            next_state, reward, done, obs,last_obs = env.step(action) 

            # save transition
            states.append(state) 
            actions.append(action) 
            rewards.append(reward) 
            # print(rewards)
            next_states.append(next_state)
            dones.append(1-done)
            old_probs.append(np.reshape(old_prob,[1,N_worker]))


            # record score and check done
            for i, (r, d) in enumerate(zip(reward, done)):
                scores[i] += r

                if d==True:
                    score_history.append(scores[i])
                    scores[i] = 0.0
                    if len(score_history) > 100:
                        del score_history[0]

            state,obs,last_obs = env.checkdone(done,next_state,last_obs,obs)

            step += 1

            if step % save_interval == 0:
                # torch.save(Net.state_dict(), model_path)
                torch.save(Net.state_dict(), 'critic7.model')
                print(score_history)
                if len(score_history)>0: 
                    train_history.append(mean(score_history))
                    print("step : {}, Average score of last 100 episode : {:.1f}".format(step, mean(score_history)))

        train(Net, optimizer, states, actions, rewards, next_states, dones, old_probs)
        torch.save(Net.state_dict(), 'critic7.model')

    

from kaggle_environments import make
main()
