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

# Create ttt directory
if not os.path.isdir('ttt'):
    os.makedirs('ttt')

# Create ttt/64 directory
if not os.path.isdir('ttt/64'):
    os.makedirs('ttt/64')

# Create ttt/128 directory
if not os.path.isdir('ttt/128'):
    os.makedirs('ttt/128')
    
# Create ttt/256 directory
if not os.path.isdir('ttt/256'):
    os.makedirs('ttt/256')
'''    
# Create ttt/512 directory
if not os.path.isdir('ttt/512'):
    os.makedirs('ttt/512')
'''

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

    def random_step(self):
        """Choose a random, unoccupied move on the board to play."""
        pos = [i for i in range(9) if self.grid[i] == 0]
        move = random.choice(pos)
        return self.step(move)

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
            Environment.STATUS_VALID_MOVE  : 0,
            Environment.STATUS_INVALID_MOVE: -1,
            Environment.STATUS_WIN         : 1,
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
        done = False
        while not done:
            # Saving log prob and rewards
            #print(state) #for step 2b
            action, logprob = select_action(policy, state)
            #print(action) #for step 2c
            state, status, done = env.play_against_random(action)
            reward = get_reward(status)
            saved_logprobs.append(logprob)
            saved_rewards.append(reward)

        R = compute_returns(saved_rewards)[0]
        running_reward += R
        
        # Gradient descent update
        finish_episode(saved_rewards, saved_logprobs, gamma)
        '''
        if i_episode % (log_interval) == 0:
            torch.save(policy.state_dict(),
                       "ttt/policy-%d.pkl" % i_episode)
        '''
        if i_episode % log_interval == 0:
            print('Episode {}\tAverage return: {:.2f}'.format(
                i_episode,
                running_reward / log_interval))
            episode_list.append(i_episode)
            avg_return = running_reward / log_interval
            avg_ret_list.append(avg_return)
            if abs(avg_return - prev_return) < 1e-3:
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
    plt.savefig("Part5a")  

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
    
def load_weights_hidden(policy, hidden, episode):
    """Load saved weights"""
    weights = torch.load("ttt/%d/policy-%d.pkl" % (hidden, episode))
    policy.load_state_dict(weights)

'''################################Part5####################################'''
def part5a():
    policy = Policy(hidden_size=128)
    env = Environment()
    train(policy, env)

def train_5b(policy, env, hidden, gamma=1.0, log_interval=1000):
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
        done = False
        while not done:
            # Saving log prob and rewards
            #print(state) #for step 2b
            action, logprob = select_action(policy, state)
            #print(action) #for step 2c
            state, status, done = env.play_against_random(action)
            reward = get_reward(status)
            saved_logprobs.append(logprob)
            saved_rewards.append(reward)

        R = compute_returns(saved_rewards)[0]
        running_reward += R
        
        # Gradient descent update
        finish_episode(saved_rewards, saved_logprobs, gamma)
        
        if i_episode % (log_interval) == 0:
            torch.save(policy.state_dict(),
                       "ttt/%d/policy-%d.pkl" % (hidden, i_episode))
        
        if i_episode % log_interval == 0:
            print('Episode {}\tAverage return: {:.2f}'.format(
                i_episode,
                running_reward / log_interval))
            episode_list.append(i_episode)
            avg_return = running_reward / log_interval
            avg_ret_list.append(avg_return)
            if abs(avg_return - prev_return) < 1e-3:
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
    plt.title("Average Return vs Episode for "+str(hidden)+" hidden units")
    plt.legend(loc=0)
    plt.savefig("Part5b_"+str(hidden))

def part5b_win(checkpoints,hidden_size):
    #checkpoints = np.arange(1000,84000,1000)
    num_games = 100
    policy = Policy(hidden_size=hidden_size)
    env = Environment()
    win_record = []
    tie_record = []
    lose_record = []
    episode_record = []
    for checkpoint in checkpoints:
        results = []
        load_weights_hidden(policy, hidden_size, checkpoint)
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
    plt.xlabel("Episode")
    plt.ylabel("Win Rate")
    plt.title("Win Rate vs Episodes")
    plt.legend(loc=0)
    plt.savefig("Part5b_%d_win" % hidden_size)
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(episode_record, tie_record, label="Tie %")
    plt.xlabel("Episode")
    plt.ylabel("Tie Rate")
    plt.title("Tie Rate vs Episodes")
    plt.legend(loc=0)
    plt.savefig("Part5b_%d_tie" % hidden_size)
    
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.plot(episode_record, lose_record, label="Lose %")
    plt.xlabel("Episode")
    plt.ylabel("Lose Rate")
    plt.title("Lose Rate vs Episodes")
    plt.legend(loc=0)
    plt.savefig("Part5b_%d_lose" % hidden_size)
        
def part5b_64():
    env = Environment()
    policy = Policy(hidden_size=64)
    train_5b(policy,env,hidden=64)

def part5b_64_perf():
    checkpoints = np.arange(1000,61000,1000)
    hidden_size = 64
    part5b_win(checkpoints, hidden_size)
    
def part5b_256():
    env = Environment()
    policy = Policy(hidden_size=256)
    train_5b(policy,env,hidden=256)
    
def part5b_256_perf():
    checkpoints = np.arange(1000,195000,1000)
    hidden_size = 256
    part5b_win(checkpoints, hidden_size)
    

def train_5c(policy, env, gamma=1.0, log_interval=1000):
    """Train policy gradient."""
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10000, gamma=0.9)
    running_reward = 0
    
    episode_list = []
    num_invalid = []
    invalid_bin_list = []
    
    prev_return = 999
    
    # Select an action from each policy state
    # Changed for loop
    for i_episode in count(1):
        saved_rewards = []
        saved_logprobs = []
        state = env.reset()
        done = False
        # Keep track of invalid moves
        invalid = 0
        invalid_binary = 0
        while not done:
            # Saving log prob and rewards
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action)
            if status == 'inv': # Record number of invalid moves in this episode
                invalid += 1
                invalid_binary = 1
            reward = get_reward(status)
            saved_logprobs.append(logprob)
            saved_rewards.append(reward)
        
        episode_list.append(i_episode)
        num_invalid.append(invalid)
        invalid_bin_list.append(invalid_binary)
        
        R = compute_returns(saved_rewards)[0]
        running_reward += R
        
        # Gradient descent update
        finish_episode(saved_rewards, saved_logprobs, gamma)
        
        if i_episode % (log_interval) == 0:
            torch.save(policy.state_dict(),
                       "ttt/policy-%d.pkl" % i_episode)
        
        if i_episode % log_interval == 0:
            print('Episode {}\tAverage return: {:.2f}'.format(
                i_episode,
                running_reward / log_interval))
            #episode_list.append(i_episode)
            avg_return = running_reward / log_interval
            #avg_ret_list.append(avg_return)
            if abs(avg_return - prev_return) < 1e-3:
                break
            prev_return = avg_return
            running_reward = 0

        if i_episode % 1 == 0: # batch_size
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(episode_list, num_invalid, label="Number of Invalid Moves")
    plt.xlabel("Episode")
    plt.ylabel("Number of Invalid Moves")
    plt.title("Number of Invalid Moves vs Episode")
    plt.legend(loc=0)
    plt.savefig("Part5c")
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(episode_list, invalid_bin_list, label="Episodes with Invalid Moves")
    plt.xlabel("Episode")
    plt.ylabel("Invalid Move")
    plt.title("Episodes Where Invalid Moves Are Made")
    plt.legend(loc=0)
    plt.savefig("Part5c_bin")


def part5c():
    policy = Policy(hidden_size=128)
    env = Environment()
    train_5c(policy,env)

def part5d():
    hidden_size = 128
    policy = Policy(hidden_size=hidden_size)
    env = Environment()
    num_games = 100
    load_weights(policy,83000)
    results = []
    text_file = open("Part5d_gamelog.txt", "w")
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
        text_file.write("Game "+str(i_game)+" result: "+str(status)+"\n")
        text_file.write("-----End of Game "+str(i_game)+"-----\n")
        results.append(status)    
    text_file.close()
    results_count = Counter(results)
    win_rate = float(results_count["win"]) / num_games
    tie_rate = float(results_count["tie"]) / num_games
    lose_rate = float(results_count["lose"]) / num_games
    print("Win Rate: "+str(win_rate*100)+"%")
    print("Tie Rate: "+str(tie_rate*100)+"%")
    print("Lose Rate: "+str(lose_rate*100)+"%")
'''################################Part5####################################'''


'''################################Part6####################################'''
def part6():
    checkpoints = np.arange(1000,84000,1000)
    num_games = 100
    hidden_size = 128
    policy = Policy(hidden_size=hidden_size)
    env = Environment()
    win_record = []
    tie_record = []
    lose_record = []
    episode_record = []
    for checkpoint in checkpoints:
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
    plt.xlabel("Episode")
    plt.ylabel("Win Rate")
    plt.title("Win Rate vs Episodes")
    plt.legend(loc=0)
    plt.savefig("Part6_win")
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(episode_record, tie_record, label="Tie %")
    plt.xlabel("Episode")
    plt.ylabel("Tie Rate")
    plt.title("Tie Rate vs Episodes")
    plt.legend(loc=0)
    plt.savefig("Part6_tie")
    
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.plot(episode_record, lose_record, label="Lose %")
    plt.xlabel("Episode")
    plt.ylabel("Lose Rate")
    plt.title("Lose Rate vs Episodes")
    plt.legend(loc=0)
    plt.savefig("Part6_lose")
'''################################Part6####################################'''

'''################################Part7####################################'''
def part7():
    hidden_size = 128
    policy = Policy(hidden_size=hidden_size)
    env = Environment()
    checkpoints = np.arange(1000,84000,1000)
    pos_0_probs = []
    pos_1_probs = []
    pos_2_probs = []
    pos_3_probs = []
    pos_4_probs = []
    pos_5_probs = []
    pos_6_probs = []
    pos_7_probs = []
    pos_8_probs = []
    for checkpoint in checkpoints:
        load_weights(policy,checkpoint)
        #load_weights_hidden(policy, hidden_size, checkpoint)
        dist = first_move_distr(policy, env)
        pos_0_probs.append(dist[0][0])
        pos_1_probs.append(dist[0][1])
        pos_2_probs.append(dist[0][2])
        pos_3_probs.append(dist[0][3])
        pos_4_probs.append(dist[0][4])
        pos_5_probs.append(dist[0][5])
        pos_6_probs.append(dist[0][6])
        pos_7_probs.append(dist[0][7])
        pos_8_probs.append(dist[0][8]) 
        
    fig0 = plt.figure()
    ax0 = fig0.add_subplot(111)
    ax0.plot(checkpoints, pos_0_probs, label="Log Prob")
    plt.xlabel("Episode")
    plt.ylabel("Log Probability")
    plt.title("Log Probability of Making First Move at Position 0")
    plt.legend(loc=0)
    plt.savefig("Part7_0")
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(checkpoints, pos_1_probs, label="Log Prob")
    plt.xlabel("Episode")
    plt.ylabel("Log Probability")
    plt.title("Log Probability of Making First Move at Position 1")
    plt.legend(loc=0)
    plt.savefig("Part7_1")
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(checkpoints, pos_2_probs, label="Log Prob")
    plt.xlabel("Episode")
    plt.ylabel("Log Probability")
    plt.title("Log Probability of Making First Move at Position 2")
    plt.legend(loc=0)
    plt.savefig("Part7_2")
    
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.plot(checkpoints, pos_3_probs, label="Log Prob")
    plt.xlabel("Episode")
    plt.ylabel("Log Probability")
    plt.title("Log Probability of Making First Move at Position 3")
    plt.legend(loc=0)
    plt.savefig("Part7_3")   
    
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    ax4.plot(checkpoints, pos_4_probs, label="Log Prob")
    plt.xlabel("Episode")
    plt.ylabel("Log Probability")
    plt.title("Log Probability of Making First Move at Position 4")
    plt.legend(loc=0)
    plt.savefig("Part7_4")
    
    fig5 = plt.figure()
    ax5 = fig5.add_subplot(111)
    ax5.plot(checkpoints, pos_5_probs, label="Log Prob")
    plt.xlabel("Episode")
    plt.ylabel("Log Probability")
    plt.title("Log Probability of Making First Move at Position 5")
    plt.legend(loc=0)
    plt.savefig("Part7_5")
    
    fig6 = plt.figure()
    ax6 = fig6.add_subplot(111)
    ax6.plot(checkpoints, pos_6_probs, label="Log Prob")
    plt.xlabel("Episode")
    plt.ylabel("Log Probability")
    plt.title("Log Probability of Making First Move at Position 6")
    plt.legend(loc=0)
    plt.savefig("Part7_6")
    
    fig7 = plt.figure()
    ax7 = fig7.add_subplot(111)
    ax7.plot(checkpoints, pos_7_probs, label="Log Prob")
    plt.xlabel("Episode")
    plt.ylabel("Log Probability")
    plt.title("Log Probability of Making First Move at Position 7")
    plt.legend(loc=0)
    plt.savefig("Part7_7")
    
    fig8 = plt.figure()
    ax8 = fig8.add_subplot(111)
    ax8.plot(checkpoints, pos_8_probs, label="Log Prob")
    plt.xlabel("Episode")
    plt.ylabel("Log Probability")
    plt.title("Log Probability of Making First Move at Position 8")
    plt.legend(loc=0)
    plt.savefig("Part7_8")
'''################################Part7####################################'''

if __name__ == '__main__':
    import sys
    policy = Policy()
    env = Environment()
    
    #part5a()
    #part5b_64()
    #part5b_64_perf()
    #part5b_256()
    #part5b_256_perf()
    #part5c()
    #part5d()
    #part6()
    #part7()
    
    if len(sys.argv) == 1:
        # `python tictactoe.py` to train the agent
        train(policy, env)
    else:
        # `python tictactoe.py <ep>` to print the first move distribution
        # using weightt checkpoint at episode int(<ep>)
        ep = int(sys.argv[1])
        load_weights(policy, ep)
        print(first_move_distr(policy, env))
    
    