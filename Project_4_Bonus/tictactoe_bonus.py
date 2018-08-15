# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 23:40:26 2018

@author: Joshl
"""

from __future__ import print_function
from collections import defaultdict
from itertools import count
import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions
from torch.autograd import Variable
import matplotlib.pyplot as plt
from collections import Counter
import os

RAND_SEED = 7
torch.manual_seed(RAND_SEED)
random.seed(RAND_SEED)

# Create ttt directory for part 1
if not os.path.isdir('ttt'):
    os.makedirs('ttt')
    
# Create ttt2 directory for part 2
if not os.path.isdir('ttt2'):
    os.makedirs('ttt2')

class Environment(object):
    """
    The Tic-Tac-Toe Environment
    """
    # possible ways to win
    win_set = frozenset([(0,1,2), (3,4,5), (6,7,8), # horizontal
                         (0,3,6), (1,4,7), (2,5,8), # vertical
                         (0,4,8), (2,4,6)])         # diagonal
    # statuses
    STATUS_VALID_MOVE = 'valid'
    STATUS_INVALID_MOVE = 'inv'
    STATUS_WIN = 'win'
    STATUS_TIE = 'tie'
    STATUS_LOSE = 'lose'
    STATUS_DONE = 'done'

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the game to an empty board."""
        self.grid = np.array([0] * 9) # grid
        self.turn = 1                 # whose turn it is
        self.done = False             # whether game is done
        return self.grid

    def render(self):
        """Print what is on the board."""
        map = {0:'.', 1:'x', 2:'o'} # grid label vs how to plot
        print(''.join(map[i] for i in self.grid[0:3]))
        print(''.join(map[i] for i in self.grid[3:6]))
        print(''.join(map[i] for i in self.grid[6:9]))
        print('====')

    def check_win(self):
        """Check if someone has won the game."""
        for pos in self.win_set:
            # s would be all 1 if all positions of a winning move is fulfilled
            # otherwise 1s and 0s
            s = set([self.grid[p] for p in pos])
            if len(s) == 1 and (0 not in s):
                return True
        return False

    def step(self, action):
        """Mark a point on position action."""
        assert type(action) == int and action >= 0 and action < 9
        # done = already finished the game
        if self.done:
            return self.grid, self.STATUS_DONE, self.done
        # action already have something on it
        if self.grid[action] != 0:
            return self.grid, self.STATUS_INVALID_MOVE, self.done
        # play move
        self.grid[action] = self.turn
        if self.turn == 1:
            self.turn = 2
        else:
            self.turn = 1
        # check win
        if self.check_win():
            self.done = True
            return self.grid, self.STATUS_WIN, self.done
        # check tie
        if all([p != 0 for p in self.grid]):
            self.done = True
            return self.grid, self.STATUS_TIE, self.done
        return self.grid, self.STATUS_VALID_MOVE, self.done
    
    def step_alt(self, action):
        """
        Mark a point on position action.
        Used when opponent moves first
        Makes sure opponent uses 'o' or 2 and we use 'x' or '1'
        """
        assert type(action) == int and action >= 0 and action < 9
        # done = already finished the game
        if self.done:
            return self.grid, self.STATUS_DONE, self.done
        # action already have something on it
        if self.grid[action] != 0:
            return self.grid, self.STATUS_INVALID_MOVE, self.done
        # play move
        if self.turn == 1: # Opponent's turn
            self.grid[action] = 2 # Makes sure opponent uses 2 or 'o'
        else: # Our turn
            self.grid[action] = 1 # Makes sure we use 1 or 'x'
        if self.turn == 1:
            self.turn = 2
        else:
            self.turn = 1
        # check win
        if self.check_win():
            self.done = True
            return self.grid, self.STATUS_WIN, self.done
        # check tie
        if all([p != 0 for p in self.grid]):
            self.done = True
            return self.grid, self.STATUS_TIE, self.done
        return self.grid, self.STATUS_VALID_MOVE, self.done

    def random_step(self):
        """Choose a random, unoccupied move on the board to play."""
        pos = [i for i in range(9) if self.grid[i] == 0]
        move = random.choice(pos)
        return self.step(move)
    
    def random_step_alt(self):
        """
        Choose a random, unoccupied move on the board to play.
        Used when opponent moves first
        """
        pos = [i for i in range(9) if self.grid[i] == 0]
        move = random.choice(pos)
        return self.step_alt(move)

    def play_against_random(self, action):
        """Play a move, and then have a random agent play the next move."""
        state, status, done = self.step(action)
        if not done and self.turn == 2:
            state, s2, done = self.random_step()
            if done:
                if s2 == self.STATUS_WIN:
                    status = self.STATUS_LOSE
                elif s2 == self.STATUS_TIE:
                    status = self.STATUS_TIE
                else:
                    raise ValueError("???")
        return state, status, done
    
    def random_move_first(self):
        """Random agent plays first."""
        state, opp_status, done = self.random_step_alt()
        status = self.STATUS_VALID_MOVE # Assign a default no reward status to our agent
        if done:
            if opp_status == self.STATUS_WIN:
                status = self.STATUS_LOSE
            elif opp_status == self.STATUS_TIE:
                status = self.STATUS_TIE
            else:
                raise ValueError("???")
        return state, status, done
    
    def play_against_random_alt(self, action):
        """
        Used after opponent makes veryfirst move
        turn 1 will be opponent so they use 'o'
        turn 2 will be ours so we use 'x'
        """
        state, status, done = self.step_alt(action)
        if not done and self.turn == 1:
            state, s2, done = self.random_step_alt()
            if done:
                if s2 == self.STATUS_WIN:
                    status = self.STATUS_LOSE
                elif s2 == self.STATUS_TIE:
                    status = self.STATUS_TIE
                else:
                    raise ValueError("???")
        return state, status, done
    
    def play_against_trained(self, policy, action):
        """
        For part2, playing against trained policy. Only updating policy for one player
        """
        state, status, done = self.step(action)
        if not done and self.turn == 2:
            state, s2, done = self.trained_opponent_action(policy, state)
            if done:
                if s2 == self.STATUS_WIN:
                    status = self.STATUS_LOSE
                elif s2 == self.STATUS_TIE:
                    status = self.STATUS_TIE
                else:
                    raise ValueError("???")
        return state, status, done
    
    def trained_opponent_action(self, policy, state):
        """ Returns action for trained opponent in part 2 """
        action, logprob = select_action(policy, state)
        state, status, done = self.step(action)
        return state, status, done
    
    def trained_move_first(self,policy,state):
        """Random agent plays first."""
        state, opp_status, done = self.trained_opponent_action_alt(policy,state)
        status = self.STATUS_VALID_MOVE # Assign a default no reward status to our agent
        if done:
            if opp_status == self.STATUS_WIN:
                status = self.STATUS_LOSE
            elif opp_status == self.STATUS_TIE:
                status = self.STATUS_TIE
            else:
                raise ValueError("???")
        return state, status, done
    
    def trained_opponent_action_alt(self, policy, state):
        """ 
        Returns action for trained opponent in part 2 when opponent starts first
        """
        action, logprob = select_action(policy, state)
        state, status, done = self.step_alt(action)
        return state, status, done
    
    def play_against_trained_alt(self, policy, action):
        """
        For part2, playing against trained policy. Only updating policy for one player
        """
        state, status, done = self.step_alt(action)
        if not done and self.turn == 2:
            state, s2, done = self.trained_opponent_action_alt(policy, state)
            if done:
                if s2 == self.STATUS_WIN:
                    status = self.STATUS_LOSE
                elif s2 == self.STATUS_TIE:
                    status = self.STATUS_TIE
                else:
                    raise ValueError("???")
        return state, status, done

class Policy(nn.Module):
    """
    The Tic-Tac-Toe Policy
    """
    def __init__(self, input_size=27, hidden_size=64, output_size=9):
        super(Policy, self).__init__()
        # TODO
        self.classifier = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        # TODO
        out = self.classifier(x)
        out = F.sigmoid(out)
        return out

def select_action(policy, state):
    """Samples an action from the policy at the state."""
    #torch.manual_seed(RAND_SEED) # Seed here is causing kernel to crash
    #print(state)
    state = torch.from_numpy(state).long().unsqueeze(0)
    state = torch.zeros(3,9).scatter_(0,state,1).view(1,27)
    #print(state) # for 2b
    pr = policy(Variable(state))
    #print(pr) # for 2c
    m = torch.distributions.Categorical(pr)
    action = m.sample()
    log_prob = torch.sum(m.log_prob(action))
    return action.data[0], log_prob

def compute_returns(rewards, gamma=1.0):
    """
    Compute returns for each time step, given the rewards
      @param rewards: list of floats, where rewards[t] is the reward
                      obtained at time step t
      @param gamma: the discount factor
      @returns list of floats representing the episode's returns
          G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ... 

    >>> compute_returns([0,0,0,1], 1.0)
    [1.0, 1.0, 1.0, 1.0]
    >>> compute_returns([0,0,0,1], 0.9)
    [0.7290000000000001, 0.81, 0.9, 1.0]
    >>> compute_returns([0,-0.5,5,0.5,-10], 0.9)
    [-2.5965000000000003, -2.8850000000000002, -2.6500000000000004, -8.5, -10.0]
    """
    returns = []
    time_step = len(rewards)
    for i in range(time_step):
        curr_ret = 0
        for j in range(i,time_step):
            curr_ret += rewards[j] * gamma**(j-i)
        returns.append(curr_ret)
    return returns

def finish_episode(saved_rewards, saved_logprobs, gamma=1.0):
    """
    Samples an action from the policy at the state.
    """
    policy_loss = []
    returns = compute_returns(saved_rewards, gamma)
    returns = torch.Tensor(returns)
    # subtract mean and std for faster training
    returns = (returns - returns.mean()) / (returns.std() +
                                            np.finfo(np.float32).eps)
    # multiplying log prob and rewards for each time state
    # sum (gamma_t * G_t) * log (pi_theta (a_t | s_t))
    for log_prob, reward in zip(saved_logprobs, returns):
        policy_loss.append(-log_prob * reward)
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward(retain_graph=True)
    # note: retain_graph=True allows for multiple calls to .backward()
    # in a single step

def get_reward(status):
    """Returns a numeric given an environment status."""
    return {
            Environment.STATUS_VALID_MOVE  : 0, # TODO
            Environment.STATUS_INVALID_MOVE: -1,
            Environment.STATUS_WIN         : 2,
            Environment.STATUS_TIE         : -0.5,
            Environment.STATUS_LOSE        : -1
    }[status]

def train(policy, env, gamma=1.0, log_interval=1000):
    """Train policy gradient."""
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10000, gamma=0.9)
    running_reward = 0
    
    #num_episodes = 70000
    episode_list = []
    avg_ret_list = []
    
    prev_return = 999
    
    # Select an action from each policy state
    # Changed for loop
    for i_episode in count(1):
        saved_rewards = []
        saved_logprobs = []
        state = env.reset()
        rand_num = random.uniform(0, 1)
        done = False
        if rand_num < 0.5:
            # We move first
            while not done:
                action, logprob = select_action(policy, state)
                state, status, done = env.play_against_random(action)
                reward = get_reward(status)
                saved_logprobs.append(logprob)
                saved_rewards.append(reward)
        else:
            # Opponent moves first
            state, status, done = env.random_move_first()
            while not done:
                action, logprob = select_action(policy, state)
                state, status, done = env.play_against_random_alt(action)
                reward = get_reward(status)
                saved_logprobs.append(logprob)
                saved_rewards.append(reward)

        R = compute_returns(saved_rewards)[0]
        running_reward += R
        
        # Gradient descent update
        #print("length of rewards: "+str(len(saved_rewards)))
        #print("length of log probs: "+str(len(saved_logprobs)))
        finish_episode(saved_rewards, saved_logprobs, gamma)
        
        if i_episode % (log_interval) == 0:
            torch.save(policy.state_dict(),
                       "ttt/policy-%d.pkl" % i_episode)
        
        if i_episode % log_interval == 0:
            print('Episode {}\tAverage return: {:.2f}'.format(
                i_episode,
                running_reward / log_interval))
            episode_list.append(i_episode)
            avg_return = running_reward / log_interval
            avg_ret_list.append(avg_return)
            if abs(avg_return - prev_return) < 1e-4:
                break
            prev_return = avg_return
            running_reward = 0

        if i_episode % 1 == 0: # batch_size
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(episode_list, avg_ret_list, label="Average Return")
    plt.xlabel("Episode")
    plt.ylabel("Average Return")
    plt.title("Average Return vs Episode")
    plt.legend(loc=0)
    plt.savefig("Part1_train_curve")  

def first_move_distr(policy, env):
    """Display the distribution of first moves."""
    state = env.reset()
    state = torch.from_numpy(state).long().unsqueeze(0)
    state = torch.zeros(3,9).scatter_(0,state,1).view(1,27)
    pr = policy(Variable(state))
    return pr.data


def load_weights(policy, episode):
    """Load saved weights"""
    weights = torch.load("ttt/policy-%d.pkl" % episode)
    policy.load_state_dict(weights)

def load_weights_part2(policy, episode):
    """Load saved weights for part 2"""
    weights = torch.load("ttt2/policy-%d.pkl" % episode)
    policy.load_state_dict(weights)

""" ########## Part 1 ########## """
def part1_train():
    """
    Using
    Environment.STATUS_VALID_MOVE  : 0,
    Environment.STATUS_INVALID_MOVE: -1,
    Environment.STATUS_WIN         : 2,
    Environment.STATUS_TIE         : -0.5,
    Environment.STATUS_LOSE        : -1
    128
    Win Rate: 85.0%
    Tie Rate: 4.0%
    Lose Rate: 11.0%
    Win Rate: 51.0%
    Tie Rate: 6.0%
    Lose Rate: 43.0%
    """
    policy = Policy(hidden_size=128)
    env = Environment()
    train(policy, env)

def part1_games_first():
    hidden_size = 128
    policy = Policy(hidden_size=hidden_size)
    env = Environment()
    num_games = 100
    load_weights(policy,184000)
    #load_weights(policy,64000)
    results = []
    text_file = open("Part1_first_gamelog.txt", "w")
    for i_game in range(num_games):
        text_file.write("Start of Game "+str(i_game+1)+"\n")
        state = env.reset()
        done = False
        move_num = 0
        while not done:
            # Saving log prob and rewards
            text_file.write("Game "+str(i_game+1)+", Turn: "+str(move_num+1)+"\n")
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action)
            map = {0:'.', 1:'x', 2:'o'} # grid label vs how to plot
            a = ''.join(map[i] for i in env.grid[0:3])
            b = ''.join(map[i] for i in env.grid[3:6])
            c = ''.join(map[i] for i in env.grid[6:9])
            d = '===='
            text_file.write(a+'\n')
            text_file.write(b+'\n')
            text_file.write(c+'\n')
            text_file.write(d+'\n')
            text_file.write("\n")
            move_num += 1
        text_file.write("Game "+str(i_game+1)+" result: "+str(status)+"\n")
        text_file.write("-----End of Game "+str(i_game+1)+"-----\n")
        results.append(status)    
    text_file.close()
    results_count = Counter(results)
    win_rate = float(results_count["win"]) / num_games
    tie_rate = float(results_count["tie"]) / num_games
    lose_rate = float(results_count["lose"]) / num_games
    print("Win Rate: "+str(win_rate*100)+"%")
    print("Tie Rate: "+str(tie_rate*100)+"%")
    print("Lose Rate: "+str(lose_rate*100)+"%")
    
def part1_games_second():
    hidden_size = 128
    policy = Policy(hidden_size=hidden_size)
    env = Environment()
    num_games = 100
    load_weights(policy,184000)
    #load_weights(policy,64000)
    results = []
    text_file = open("Part1_second_gamelog.txt", "w")
    for i_game in range(num_games):
        text_file.write("Start of Game "+str(i_game+1)+"\n")
        state = env.reset()
        done = False
        move_num = 0
        # Opponent moves first
        state, status, done = env.random_move_first()
        # Record Opponent First Move
        text_file.write("Game "+str(i_game+1)+", Turn: "+str(move_num)+"\n")
        map = {0:'.', 1:'x', 2:'o'} # grid label vs how to plot
        a = ''.join(map[i] for i in env.grid[0:3])
        b = ''.join(map[i] for i in env.grid[3:6])
        c = ''.join(map[i] for i in env.grid[6:9])
        d = '===='
        text_file.write(a+'\n')
        text_file.write(b+'\n')
        text_file.write(c+'\n')
        text_file.write(d+'\n')
        text_file.write("\n")
        while not done:
            # Saving log prob and rewards
            text_file.write("Game "+str(i_game+1)+", Turn: "+str(move_num+1)+"\n")
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random_alt(action)
            map = {0:'.', 1:'x', 2:'o'} # grid label vs how to plot
            a = ''.join(map[i] for i in env.grid[0:3])
            b = ''.join(map[i] for i in env.grid[3:6])
            c = ''.join(map[i] for i in env.grid[6:9])
            d = '===='
            text_file.write(a+'\n')
            text_file.write(b+'\n')
            text_file.write(c+'\n')
            text_file.write(d+'\n')
            text_file.write("\n")
            move_num += 1
        text_file.write("Game "+str(i_game+1)+" result: "+str(status)+"\n")
        text_file.write("-----End of Game "+str(i_game+1)+"-----\n")
        results.append(status)    
    text_file.close()
    results_count = Counter(results)
    win_rate = float(results_count["win"]) / num_games
    tie_rate = float(results_count["tie"]) / num_games
    lose_rate = float(results_count["lose"]) / num_games
    print("Win Rate: "+str(win_rate*100)+"%")
    print("Tie Rate: "+str(tie_rate*100)+"%")
    print("Lose Rate: "+str(lose_rate*100)+"%")

def part1_track_win_first():
    checkpoints = np.arange(1000,185000,1000)
    #checkpoints = np.arange(1000,65000,1000)
    num_games = 100
    hidden_size = 128
    policy = Policy(hidden_size=hidden_size)
    env = Environment()
    win_record = []
    tie_record = []
    lose_record = []
    episode_record = []
    for checkpoint in checkpoints:
        #print(checkpoint)
        results = []
        load_weights(policy,checkpoint)
        #load_weights_hidden(policy, hidden_size, checkpoint)
        for i_game in range(num_games):  
            done = False
            state = env.reset()
            while not done:
                # Saving log prob and rewards
                action, logprob = select_action(policy, state)
                state, status, done = env.play_against_random(action)
            results.append(status)
        results_count = Counter(results)
        win_rate = (float(results_count["win"]) / num_games)*100
        tie_rate = (float(results_count["tie"]) / num_games)*100
        lose_rate = (float(results_count["lose"]) / num_games)*100
        win_record.append(win_rate)
        tie_record.append(tie_rate)
        lose_record.append(lose_rate)
        episode_record.append(checkpoint)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(episode_record, win_record, label="Win %")
    ax1.plot(episode_record, tie_record, 'g-',label="Tie %")
    ax1.plot(episode_record, lose_record,'r-', label="Lose %")
    plt.xlabel("Episode")
    plt.ylabel("Win/Tie/Lose Rate")
    plt.title("Win/Tie/Lose Rate vs Episodes Starting First")
    plt.legend(loc=0)
    plt.savefig("Part1_first_win")
    
    
def part1_track_win_second():
    checkpoints = np.arange(1000,185000,1000)
    num_games = 100
    hidden_size = 128
    policy = Policy(hidden_size=hidden_size)
    env = Environment()
    win_record = []
    tie_record = []
    lose_record = []
    episode_record = []
    for checkpoint in checkpoints:
        #print(checkpoint)
        results = []
        load_weights(policy,checkpoint)
        #load_weights_hidden(policy, hidden_size, checkpoint)
        for i_game in range(num_games):  
            done = False
            state = env.reset()
            state, status, done = env.random_move_first()
            while not done:
                action, logprob = select_action(policy, state)
                state, status, done = env.play_against_random_alt(action)
            results.append(status)
        results_count = Counter(results)
        win_rate = (float(results_count["win"]) / num_games)*100
        tie_rate = (float(results_count["tie"]) / num_games)*100
        lose_rate = (float(results_count["lose"]) / num_games)*100
        win_record.append(win_rate)
        tie_record.append(tie_rate)
        lose_record.append(lose_rate)
        episode_record.append(checkpoint)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(episode_record, win_record, label="Win %")
    ax1.plot(episode_record, tie_record, 'g-', label="Tie %")
    ax1.plot(episode_record, lose_record, 'r-', label="Lose %")
    plt.xlabel("Episode")
    plt.ylabel("Win/Tie/Lose Rate")
    plt.title("Win/Tie/Lose Rate vs Episodes Starting Second")
    plt.legend(loc=0)
    plt.savefig("Part1_second_win")
    
""" ########## Part 1 ########## """

""" ########## Part 2 ########## """

def train_part2(policy, env, gamma=1.0, log_interval=1000):
    """Train policy gradient."""
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10000, gamma=0.9)
    running_reward = 0
    
    running_reward_rand = 0
    
    #num_episodes = 70000
    episode_list = []
    avg_ret_list = []
    
    avg_ret_list_rand = []
    
    prev_return = 999
    
    # Select an action from each policy state
    # Changed for loop
    for i_episode in count(1):
        saved_rewards = []
        saved_logprobs = []
        
        saved_rewards_rand = []
        saved_logprobs_rand = []
        
        state = env.reset()
        done = False
        done_rand = False
        rand_num = random.uniform(0, 1)
        if rand_num < 0.25:
            # We move first against trained policy
            while not done:
                action, logprob = select_action(policy, state)
                state, status, done = env.play_against_trained(policy, action)
                reward = get_reward(status)
                saved_logprobs.append(logprob)
                saved_rewards.append(reward)
            
            state = env.reset()  
            while not done_rand:
                action, logprob = select_action(policy, state)
                #print(action) #for step 2c
                state_rand, status_rand, done_rand = env.play_against_random(action)
                #print(status2)
                reward_rand = get_reward(status_rand)
                saved_logprobs_rand.append(logprob)
                saved_rewards_rand.append(reward_rand)
        elif rand_num < 0.5:
            # Trained opponent moves first
            state, status, done = env.trained_move_first(policy,state)
            while not done:
                action, logprob = select_action(policy, state)
                state, status, done = env.play_against_trained_alt(policy,action)
                reward = get_reward(status)
                saved_logprobs.append(logprob)
                saved_rewards.append(reward)
            
            state = env.reset()
            state_rand, status_rand, done_rand = env.random_move_first()
            while not done_rand:
                action, logprob = select_action(policy, state_rand)
                state_rand, status_rand, done_rand = env.play_against_random_alt(action)
                reward_rand = get_reward(status_rand)
                saved_logprobs_rand.append(logprob)
                saved_rewards_rand.append(reward_rand)
        elif rand_num < 0.75:
            # We move first against random
            while not done:
                action, logprob = select_action(policy, state)
                state, status, done = env.play_against_random(action)
                reward = get_reward(status)
                saved_logprobs.append(logprob)
                saved_rewards.append(reward)
                
                saved_logprobs_rand.append(logprob)
                saved_rewards_rand.append(reward)
        else:
            # Random opponent moves first
            state, status, done = env.random_move_first()
            while not done:
                action, logprob = select_action(policy, state)
                state, status, done = env.play_against_random_alt(action)
                reward = get_reward(status)
                saved_logprobs.append(logprob)
                saved_rewards.append(reward)
                
                saved_logprobs_rand.append(logprob)
                saved_rewards_rand.append(reward)

        R = compute_returns(saved_rewards)[0]
        running_reward += R

        R_rand = compute_returns(saved_rewards_rand)[0]
        running_reward_rand += R_rand
        
        # Gradient descent update
        finish_episode(saved_rewards, saved_logprobs, gamma)
        
        if i_episode % (log_interval) == 0:
            torch.save(policy.state_dict(),
                       "ttt2/policy-%d.pkl" % i_episode)
        
        if i_episode % log_interval == 0:
            print('Episode {}\tAverage return: {:.2f}'.format(
                i_episode,
                running_reward / log_interval))
            print('Random, Episode {}\tAverage return: {:.2f}'.format(
                i_episode,
                running_reward_rand / log_interval))
            episode_list.append(i_episode)
            avg_return = running_reward / log_interval
            avg_ret_list.append(avg_return)
            
            avg_return_rand = running_reward_rand / log_interval
            avg_ret_list_rand.append(avg_return_rand)
            
            if abs(avg_return - prev_return) < 1e-3: #1e-4 before
                break
            prev_return = avg_return
            running_reward = 0
            running_reward_rand = 0

        if i_episode % 1 == 0: # batch_size
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(episode_list, avg_ret_list, label="Average Return")
    ax1.plot(episode_list, avg_ret_list_rand, 'r-', label="Random Average Return")
    plt.xlabel("Episode")
    plt.ylabel("Average Return")
    plt.title("Average Return vs Episode")
    plt.legend(loc=0)
    plt.savefig("Part2_train_curve")

def part2_games_first():
    hidden_size = 128
    policy = Policy(hidden_size=hidden_size)
    env = Environment()
    num_games = 100
    load_weights_part2(policy,57000) #Best
    #load_weights_part2(policy,57000)
    results = []
    text_file = open("Part2_first_gamelog.txt", "w")
    for i_game in range(num_games):
        text_file.write("Start of Game "+str(i_game+1)+"\n")
        state = env.reset()
        done = False
        move_num = 0
        while not done:
            # Saving log prob and rewards
            text_file.write("Game "+str(i_game+1)+", Turn: "+str(move_num+1)+"\n")
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action)
            map = {0:'.', 1:'x', 2:'o'} # grid label vs how to plot
            a = ''.join(map[i] for i in env.grid[0:3])
            b = ''.join(map[i] for i in env.grid[3:6])
            c = ''.join(map[i] for i in env.grid[6:9])
            d = '===='
            text_file.write(a+'\n')
            text_file.write(b+'\n')
            text_file.write(c+'\n')
            text_file.write(d+'\n')
            text_file.write("\n")
            move_num += 1
        text_file.write("Game "+str(i_game+1)+" result: "+str(status)+"\n")
        text_file.write("-----End of Game "+str(i_game+1)+"-----\n")
        results.append(status)    
    text_file.close()
    results_count = Counter(results)
    win_rate = float(results_count["win"]) / num_games
    tie_rate = float(results_count["tie"]) / num_games
    lose_rate = float(results_count["lose"]) / num_games
    print("Win Rate: "+str(win_rate*100)+"%")
    print("Tie Rate: "+str(tie_rate*100)+"%")
    print("Lose Rate: "+str(lose_rate*100)+"%")
    
def part2_games_second():
    hidden_size = 128
    policy = Policy(hidden_size=hidden_size)
    env = Environment()
    num_games = 100
    load_weights_part2(policy,57000) # Best
    #load_weights_part2(policy,57000)
    results = []
    text_file = open("Part2_second_gamelog.txt", "w")
    for i_game in range(num_games):
        text_file.write("Start of Game "+str(i_game+1)+"\n")
        state = env.reset()
        done = False
        move_num = 0
        # Opponent moves first
        state, status, done = env.random_move_first()
        # Record Opponent First Move
        text_file.write("Game "+str(i_game+1)+", Turn: "+str(move_num)+"\n")
        map = {0:'.', 1:'x', 2:'o'} # grid label vs how to plot
        a = ''.join(map[i] for i in env.grid[0:3])
        b = ''.join(map[i] for i in env.grid[3:6])
        c = ''.join(map[i] for i in env.grid[6:9])
        d = '===='
        text_file.write(a+'\n')
        text_file.write(b+'\n')
        text_file.write(c+'\n')
        text_file.write(d+'\n')
        text_file.write("\n")
        while not done:
            # Saving log prob and rewards
            text_file.write("Game "+str(i_game+1)+", Turn: "+str(move_num+1)+"\n")
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random_alt(action)
            map = {0:'.', 1:'x', 2:'o'} # grid label vs how to plot
            a = ''.join(map[i] for i in env.grid[0:3])
            b = ''.join(map[i] for i in env.grid[3:6])
            c = ''.join(map[i] for i in env.grid[6:9])
            d = '===='
            text_file.write(a+'\n')
            text_file.write(b+'\n')
            text_file.write(c+'\n')
            text_file.write(d+'\n')
            text_file.write("\n")
            move_num += 1
        text_file.write("Game "+str(i_game+1)+" result: "+str(status)+"\n")
        text_file.write("-----End of Game "+str(i_game+1)+"-----\n")
        results.append(status)    
    text_file.close()
    results_count = Counter(results)
    win_rate = float(results_count["win"]) / num_games
    tie_rate = float(results_count["tie"]) / num_games
    lose_rate = float(results_count["lose"]) / num_games
    print("Win Rate: "+str(win_rate*100)+"%")
    print("Tie Rate: "+str(tie_rate*100)+"%")
    print("Lose Rate: "+str(lose_rate*100)+"%")

def part2_games_against_trained_first():
    hidden_size = 128
    policy = Policy(hidden_size=hidden_size)
    env = Environment()
    num_games = 100
    load_weights_part2(policy,57000) #Best
    #load_weights_part2(policy,57000)
    results = []
    text_file = open("Part2_first_vs_trained_gamelog.txt", "w")
    for i_game in range(num_games):
        text_file.write("Start of Game "+str(i_game+1)+"\n")
        state = env.reset()
        done = False
        move_num = 0
        while not done:
            # Saving log prob and rewards
            text_file.write("Game "+str(i_game+1)+", Turn: "+str(move_num+1)+"\n")
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_trained(policy, action)
            map = {0:'.', 1:'x', 2:'o'} # grid label vs how to plot
            a = ''.join(map[i] for i in env.grid[0:3])
            b = ''.join(map[i] for i in env.grid[3:6])
            c = ''.join(map[i] for i in env.grid[6:9])
            d = '===='
            text_file.write(a+'\n')
            text_file.write(b+'\n')
            text_file.write(c+'\n')
            text_file.write(d+'\n')
            text_file.write("\n")
            move_num += 1
        text_file.write("Game "+str(i_game+1)+" result: "+str(status)+"\n")
        text_file.write("-----End of Game "+str(i_game+1)+"-----\n")
        results.append(status)    
    text_file.close()
    results_count = Counter(results)
    win_rate = float(results_count["win"]) / num_games
    tie_rate = float(results_count["tie"]) / num_games
    lose_rate = float(results_count["lose"]) / num_games
    print("Win Rate: "+str(win_rate*100)+"%")
    print("Tie Rate: "+str(tie_rate*100)+"%")
    print("Lose Rate: "+str(lose_rate*100)+"%")

def part2_track_win_first():
    checkpoints = np.arange(1000,58000,1000) # Best
    #checkpoints = np.arange(1000,20000,1000)
    num_games = 100
    hidden_size = 128
    policy = Policy(hidden_size=hidden_size)
    env = Environment()
    win_record = []
    tie_record = []
    lose_record = []
    episode_record = []
    for checkpoint in checkpoints:
        #print(checkpoint)
        results = []
        load_weights_part2(policy,checkpoint)
        #load_weights_hidden(policy, hidden_size, checkpoint)
        for i_game in range(num_games):  
            done = False
            state = env.reset()
            while not done:
                # Saving log prob and rewards
                action, logprob = select_action(policy, state)
                state, status, done = env.play_against_random(action)
            results.append(status)
        results_count = Counter(results)
        win_rate = (float(results_count["win"]) / num_games)*100
        tie_rate = (float(results_count["tie"]) / num_games)*100
        lose_rate = (float(results_count["lose"]) / num_games)*100
        win_record.append(win_rate)
        tie_record.append(tie_rate)
        lose_record.append(lose_rate)
        episode_record.append(checkpoint)
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(episode_record, win_record, label="Win %")
    ax1.plot(episode_record, tie_record, 'g-', label="Tie %")
    ax1.plot(episode_record, lose_record, 'r-', label="Lose %")
    plt.xlabel("Episode")
    plt.ylabel("Win Rate")
    plt.title("Win Rate vs Episodes Starting First")
    plt.legend(loc=0)
    plt.savefig("Part1_first_win")
    
def part2_track_win_second():
    checkpoints = np.arange(1000,58000,1000) #Best
    #checkpoints = np.arange(1000,20000,1000)
    num_games = 100
    hidden_size = 128
    policy = Policy(hidden_size=hidden_size)
    env = Environment()
    win_record = []
    tie_record = []
    lose_record = []
    episode_record = []
    for checkpoint in checkpoints:
        #print(checkpoint)
        results = []
        load_weights_part2(policy,checkpoint)
        #load_weights_hidden(policy, hidden_size, checkpoint)
        for i_game in range(num_games):  
            done = False
            state = env.reset()
            state, status, done = env.random_move_first()
            while not done:
                action, logprob = select_action(policy, state)
                state, status, done = env.play_against_random_alt(action)
            results.append(status)
        results_count = Counter(results)
        win_rate = (float(results_count["win"]) / num_games)*100
        tie_rate = (float(results_count["tie"]) / num_games)*100
        lose_rate = (float(results_count["lose"]) / num_games)*100
        win_record.append(win_rate)
        tie_record.append(tie_rate)
        lose_record.append(lose_rate)
        episode_record.append(checkpoint)
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(episode_record, win_record, label="Win %")
    ax1.plot(episode_record, tie_record, 'g-', label="Tie %")
    ax1.plot(episode_record, lose_record, 'r-', label="Lose %")
    plt.xlabel("Episode")
    plt.ylabel("Win Rate")
    plt.title("Win Rate vs Episodes Starting Second")
    plt.legend(loc=0)
    plt.savefig("Part2_second_win")
    

def part2_train():
    """
    Will only be updating agent 1 policy
    stopping criteria: 1e-3
    Don't start from best weights
    lr = 0.001
    gamma = 0.9
    Environment.STATUS_VALID_MOVE  : 0,
    Environment.STATUS_INVALID_MOVE: -1,
    Environment.STATUS_WIN         : 2,
    Environment.STATUS_TIE         : -0.5,
    Environment.STATUS_LOSE        : -1
    128 Use this
    Win Rate: 83.0%
    Tie Rate: 4.0%
    Lose Rate: 13.0%
    Win Rate: 63.0%
    Tie Rate: 3.0%
    Lose Rate: 34.0%   
    
    """
    policy = Policy(hidden_size=128)
    env = Environment()
    # Load best weights from part 1
    #load_weights(policy,184000) #Final weight from part 1
    train_part2(policy, env)

""" ########## Part 2 ########## """

if __name__ == '__main__':
    part1_train()
    part1_games_first()
    part1_games_second()
    part1_track_win_first()
    part1_track_win_second()
    part2_train()
    part2_games_first()
    part2_games_second()
    part2_track_win_first()
    part2_track_win_second()
    part2_games_against_trained_first()
