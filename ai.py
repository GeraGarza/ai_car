# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os  # save/load brain
import torch  # good for ai, can handle dynamic graphs
import torch.nn as nn # neural networks | Deep NN 3 signals, 3 orientations -> action to play
import torch.nn.functional as F  # contains different functions, loss funciton (relu  loss)
import torch.optim as optim  # optimizer to performa stochastic gradient descent
import torch.autograd as autograd # convert from tensor (advanced array) to variable w gradient
from torch.autograd import Variable


# Creating the architecture of the NN
class Network(nn.Module):
    
    # input size = 5 (3 senosrs + 2 orientations[srt, dst])
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30) # fc - full connection from neruons of input layer to hidden layer
        self.fc2 = nn.Linear(30, nb_action)
        
        
    def forward(self, state):
        # x hidden neurons = use fc1 to get hidden neurons then apply relu; then input our input states
        x = F.relu(self.fc1(state)) # rectified function
        q_values = self.fc2(x)
        return q_values
    
    
    # Implement Experience Replay
    # Learn long term correlations
    # put the last X transitons intow memory
    
class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        
    def push(self, event):
        # tuple 4 elements (last state, new state , last action, last reward)
        #                  (st, st+1, at, rt)
        self.memory.append(event) 
        
        if len(self.memory) > self.capacity:
            del self.memory[0]
        
        
    # get random samples from our memory
    def sample(self, batch_size):
        # take mem; use batchsize; pytorch to get format of samples
        #my_list = [[1,2,3],[4,5,6]] 
        #*zip(*my_list) --> (1, 4) (2, 5) (3, 6)
        samples = zip(*random.sample(self.memory, batch_size))
        
        # Variable --> convert torch tensor to Torch variable (tensor and gradient)
        # for each batch contained in sample (ex: batch of actions), concate to 1st dim (ea correspond to same time t)
        # each row (state action reward) corr to t | evetually we get a list of batches all aligned and each is a torch var
        return map(lambda x: Variable(torch.cat(x,0)), samples)
        
        
# Implement Deep Q Learning
    
class Dqn():
    
    #  will need: dnn | memory | last state | last rewards | optimizer (perform stochastic gradient descent )           
    #  Dqn(5,3,0.9)
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []  # mean of last 100 rewards, used to eval evloution of performance
        self.model = Network(input_size, nb_action)
        #  100k transitions
        self.memory = ReplayMemory(100_000)
        self.optimizer = optim.Adam(self.model.parameters(),lr = 1e-2)
        # 5-dim vector (encoding state of env) 3 signlas (left straigt right) orientation
        # needs to be a torch tensor + fake dimenstion (batch) | network can only accpt batch of input observations
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
        
        
        
        
    #  input of nn is the state (5 dim)
    def select_action(self, state):
        #  funcitonal -> input the entities for which we want to gen a prob distribution (q-values)
        #  (q-values) = output of nn | want output | state needs to be torch tensor since we will use
        #  self.last_state for select_action, also wont be using gradient 
        #  Temperature(hyperparameter) increases the sensitivity to low probability candidates.
        #  softmax([1,2,3]) [0.04,0.11,0.85] * 3 =  [0,0.2,0.98]
        #  Tempurate = 7
        t = 0
        probs = F.softmax(self.model(Variable(state, volatile = True))*t)
        
        #  take random draw from distribution to get final action
        action = probs.multinomial()
        #  action has a fake batch, must return only action
        return action.data[0,0]
        
        
        
    #  train dnn (inside ai) forward propigation then back prop, get output and target and compare w target
    #  computer loss error, back prop loss err, using stochasitc gradient desent 
    #  we will update weights accroding to infulence on loss err
    #  they are all alighned respect to time, due to concatination
    #  if we only had tranistions by themselves, it would be instant learning and not learn (short learning)
    #  need to take batches, use different outputs
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        
        # first get the output of the batch state
        # model is expecting batch as input
        # we dont want all possible actions [0,1,2] | we want actions that were chosen tfr use gather 
        # batch actions need same dimentions | 0 = fake dim of state, 1 = fake dim of action 
        # then kill fake batch w sqeeze, since we are out of nn, we dont want in batch, we want in simple tesor
        # batch is only when we work in nn, tfr kill using sqeeze
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).sqeeze(1)
        
        # dql process  
        # initialize q values
        # each time t, select action function | the option with softmax
        # then append the transition, get prediction, get target, compute loss
        # target needs next output, target = gamma * next output + reward
        # next output = result of nn, input is the batch next state, but now next output = max of q values, wrt all actions
        # detach, to detach all outputs of model, then take max of all q-values, it has to be wrt to action[1]
        # then we want q-val of st+1, next state[0]
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        
        # target = reward + gamma  * next_output
        target = self.gamma * next_outputs + batch_reward
        
        # can compute loss (temperal difference) | error of pred
        # huber loss function is a good loss func
        td_loss = F.smooth_l1_loss(outputs, target)
        
        # now we can back prop error back into network using stochasict gradient descent to update weights
        # need to reinitalize optimizer at each iteration of loop
        self.optimizer.zero_grad()
        
        # now perform back prop
        # retain_variables will freeze mem, will imporve training performance
        td_loss.backward(retain_variables = True)
        
        # update weights according to back prop, depending on contribution to error
        self.optimizer.step() 
        
        
        
        
    #  update function, as soon as ai reaches new state
    #  last action/state/rew = new action/state/rew
    #  append transitions, update reward window (exploration)
    #  update will update our ai and return next action to play
    #  reward is the last reward from prev move (went to sand etc.)
    def update(self, reward, new_signal):
        
        # update elems of transitions
        # update new state, depends on signal (signal = state) 
        # get reflex of what we need to do next, create fake dim (corresponding to batch) | index of fake dim = 0
        # state is input of nn tfr convert to torch sensor
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        
        # signals are the density of scent detected around sensors
        # at each time t, the transition is composed of cur_state (st), next_state(st+1), reward(rt), action(at)
        # just got last element for transition (st+1), brand new tranistion to mem
        
        
        # now update memory, by getting new state, we are getting new transition, then append to mem
        # convert everything to torch tensor, including last_action
        # push functions appends transition(event) to mem
        # push_transition (last state, new_state, last_action(long tensor of one ele), last_reward)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        
        # we need to play an action | use select action
        # takes state as input, returns output of nn 
        action = self.select_action(new_state)
        
        # get reward, if we have a full memory, start learning
        # num of trans we want ai to learn from
        trans = 100
        if len(self.memory.memory) > trans:
            batch_state, batch_next_state, batch_reward, batch_action = self.memory.sample(trans)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
            
        # update last variables
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        
        # taking avg of last 1000 rewards, to see the evolution of rewards
        self.reward_window.append(reward)
        
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
            
        return action
        
        
        
        
    #  compute score (mean) from sliding window
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window) +1.)
        
    #  optimizer holds the last weights of iteration
    def save(self):
        torch.save({'state_dict':self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                    }, 'last_brain.pth')
        
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print('=> loading checkpoint... ')
            checkpoint = torch.load('last_brain.pth')
            # update our existing model and opt with checkpoints
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print('done')
        else:
            print('no such file')
        
            
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        