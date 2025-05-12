
# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.
import numpy as np
import random
import matplotlib
from matplotlib import pyplot as plt

from RPS_game import play, mrugesh, abbey, quincy, kris, human, random_player
from unittest import main

STATES = {s:id for id, s in enumerate(a for a in 'RPS')}
ACTIONS = {0:'R', 1:'P', 2:'S'}

# initialize q_matrix with zeros
q_matrix = np.ones((len(STATES), len(ACTIONS)))*10

# dict to see if won or not for reward calculation
loss_dict = {'R':'S', 'P':'R', 'S':'P'} # init with 'S' (arbitrary)

# List of rewards
rewards = [1]


# EPISODES = 500
MAX_STEPS = 60
LEARNING_RATE = 0.1 # LEARNING_RATE higher means more drastic changes
GAMMA = 1. # higher equals more weight on next steps' rewards

def softmax(X, temp):
    exp_X = np.exp(np.round((X - np.max(X))/temp, 2)) # Normalize to avoid infinity values
    return exp_X / np.sum(exp_X)

def player(prev_play, opponent_history=[]):
    '''
    Function to get an action (Rock, Paper, or Scissors) given a previous
    action by opponent
    Parameters:
    -----------
    prev_play : str, element of {'R', 'P', 'S'}
    opponent_history : list, default empty list
    '''
    global q_matrix, loss_dict, MAX_STEPS, ACTIONS, \
        STATES, rewards, LEARNING_RATE#, o_bars , epsilon


    # Initialize opponent history and epsilon at first
    if not prev_play:
        q_matrix = q_matrix*0 + 1 # revert q_matrix to init_state if new bot
        temp = 1
        rewards = [0]
        state_id = 1 # initialize at tie
        prev_play = 'R' # to avoid breaking, initialize with R
        action_id = 1 # Play 'P' first time
        action_id = random.choice(range(3))
        opponent_history = [(prev_play, int(action_id))]
        return ACTIONS[action_id] # Start randomly
    
    state_id = STATES[prev_play]    

    rewards_np = np.array(rewards)
    loss_rate = np.sum(rewards_np == -1) / len(rewards)
    win_rate = np.sum(rewards_np == 1) / len(rewards)
    if win_rate - loss_rate > 0.2:
        temp = 0.6
        LEARNING_RATE = 0.01
    else:
        temp = loss_rate + 0.6 # inc temp if losing more
        LEARNING_RATE = 0.1 

    probs = softmax(q_matrix[state_id, :], temp)

    
    if np.sum(np.isnan(probs)) > 0:
        print(q_matrix[state_id])
        print(probs)
        X = q_matrix[state_id]
        print(X - np.max(X))
        print(np.exp((X - np.max(X))/0.9))
        print(np.sum(np.exp(X - np.max(X))), temp)
    action_id = np.random.choice(range(3), p=probs)
    
    # print(opponent_history)
    opponent_history.append((prev_play, int(action_id)))
    # print(opponent_history)
    if ACTIONS[action_id] == loss_dict[opponent_history[-1][0]]:
        reward = -1
    elif ACTIONS[action_id] == opponent_history[-1][0]:
        reward = 0
    else:
        reward = 1

    rewards.append(reward)

    # Add information to the opponent history memory (opp_play, observation=new_state, our_play)
    # opponent_history.append((prev_play+ACTIONS[action_id], int(action_id)))
    if len(rewards) > MAX_STEPS: # and len(rewards) % MAX_STEPS == 0:
        for i in range(MAX_STEPS, 0, -1):
            # Change MAX_STEPS back step and then proceed each time
            prev_state_id = STATES[opponent_history[-i][0]] # state index of last to last opp move
            # new_state_id = STATES[opponent_history[-1][0]] # state index of last opp move
            prev_action_id = opponent_history[-i][1] # last game what action we did
            prev_reward = rewards[-i]
            state_id = STATES[opponent_history[-i+1][0]]
            # Update q-matrix based on action and reward at end of each game using bellman's equation from FCC
            q_matrix[prev_state_id, prev_action_id] += LEARNING_RATE * (prev_reward + GAMMA * np.max(q_matrix[state_id, :]) - q_matrix[prev_state_id, prev_action_id])

    return ACTIONS[action_id]


# Plot games against abbey to see what's going on
fig, axs = plt.subplots(3, 1, figsize=(20, 25))
n_games = 1000
play(player, abbey, n_games)
# print(rewards)

axs[0].hist(rewards)
axs[0].set_title(f"Results of playing with Abbey: Average reward = {np.mean(rewards)}")

axs[1].plot(range(500), rewards[:500])
axs[1].set_title("progression of first 500 games with Abbey")


axs[2].plot(range(n_games-500, n_games), rewards[-500:])
axs[2].set_title("progression of last 500 games with Abbey")

plt.savefig("demo1.png")
