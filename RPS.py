
# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.
import numpy as np
import random
from matplotlib import pyplot as plt

from RPS_game import play, mrugesh, abbey, quincy, kris, human, random_player
from unittest import main



# #### Right problem ####
# initial state = somewhat foggy implementation of reinforcement learning with 
#                 significantly noisy performance against bots, and random performance
#                 with Abbey. 

# goal state = clear implementation of reinforcement learning with win rate of >70%
#              with all bots. clear in that it is clear which data is used, and why
#              and what games and rewards are matched together. 
#              60%-100% win rate (range)

            
# constraints = Can only play a 1000 games in test_module. 
#               Have to use q-learning
#               Have to understand what I'm doing (can't just generate with AI)
#               Efficient enough to run in reasonable time (5s per 1000 games up. bound)

# obstacles = Determining the correct states (1 or 2 as suggested by GPT?)
#             Identifying how to store states and actions 
#             Identifying _what_ to store in global variables / between calls
#             Non stationary bot strategies preventing convergence
#             Main one - bot losing more on Abbey

# ##### Breakitdown
# What information should I store at each function call?

# How can I store these information given that the function is called with 
# opponent_history=[]?
    # - current way - store in global variables and reset if condition met 
    # (prev_play = '' means new bot - so reset q table as diff strategy)
    # Can i carry over q-tables? <-- interesting idea. would it be better for
    # performance? we'll test and see ... =)

# ##### Gapanalysis
# I've created a function with q-learning, but not really clear what exact
# information I'm storing at each function call
#     - This beats 3 bots most of the time, but random win_rate/loses against
#       one bot.

# Should we have recurrence?? <-- maybe key is recurrence

#### APPROACH ####

# define constant dicts and lists outside to avoid recreation each func call
LR = 0.5 # Learning Rate 0.5 > 0.1 > 0.15 (?!)
GAMMA = 1.
temp = 1.

last_action = None
last_state = None
opponent_history = []

# EPSILON_GREEDY = False
# epsilon = 0.1

actions =  ['R', 'P', 'S'] # rock paper or scissors

two_move_states = [a+b+c+d for a in actions \
                    for b in actions \
                        for c in actions \
                            for d in actions]
# get action id's for use as indexes in q-matrix
state_ids = {s:id for id, s in enumerate(two_move_states)} # default to rock if action empty
action_ids = {s:id for id, s in enumerate(actions)}
win_dict = {'R':'P', 'P':'S', 'S':'R'} # if pred_opp == thing, play win_dct[pred] to win

# matrix to store probs of opp move 
# q_matrix[prev_opp_action, opp_action] = Pr(opp_action/prev_opp_action)
q_matrix = np.ones((len(two_move_states), len(actions)))

# list to store reward values
rewards = []

def softmax(X, temp=1):
    """Compute the softmax of array X in a numerically stable way.
    Parameters:
        X (numpy.ndarray): Input array.
        temp (float): softening factor
    Returns:
        numpy.ndarray: Softmax probabilities of the input array.
    """
    exp_X = np.exp((X - np.max(X))/temp) # removing max befr exponent => no inf => no Nan
    return exp_X / np.sum(exp_X)


def get_rewards(opp_action, my_action):
    global loss_dict

    if opp_action == '':
        return 0
    if my_action == opp_action or opp_action == '':
        return 0 # Tie
    elif my_action == win_dict[opp_action]:
        return 1 # Win
    else:
        return -1 # Loss


def player(prev_play):
    """Give a value in response to previous play by opponent in a game of 
       rock, paper, scissors using q-learning to predict opponent strategy.

        Choose action by sampling with prob(action) = softmax(action_value)

        Update values of actions by storing 

    Parameters:
        prev_play (str, in 'RPS): Opponent's play for previous round.
    Returns:
        action (str, in 'RPS'): Our action in response to opp's next
                                predicted move.
    """
    global last_action, last_state, q_matrix, opponent_history, actions, LR, GAMMA, temp

    if prev_play == '':
        last_state = ''.join(np.random.choice(actions, size=4)) # RRP gives consistently better resuults than 'RPS
        print(last_state)
        last_action = np.random.choice(actions)
        # opponent_history = []
        q_matrix = q_matrix*0 + 1
        return last_action

    # curr_state = last_state[-3:-1] + prev_play + last_action #4value states
    curr_state = last_state[-2:] + prev_play + last_action # 3 value states (pev_prev_opp play, prev_prev_action, prev_opp play, prev_action)
    reward = get_rewards(prev_play, last_action)

    rewards.append(reward) # for plotting

    s = state_ids[last_state]
    a = action_ids[last_action]

    s_next = state_ids[curr_state]

    q_matrix[s, a] += LR * (
        reward + GAMMA * np.max(q_matrix[s_next, :]) - q_matrix[s, a]
    )

    # opponent_history.append(prev_play)

    action = np.random.choice(actions, p=softmax(q_matrix[s_next], temp))

    last_action = action
    last_state = curr_state

    return action

def plot_plays(players=[], n_games=1000, plot=True):
    global rewards
    n_opp = len(players)
    fig, axs = plt.subplots(3*n_opp, 1, figsize=(20, 25*n_opp))
    print(f"LR: {LR}, GAMMA={GAMMA}, INIT_STATE=RRPS")
    for i, opponent in enumerate(players):
        rewards = []
        play(player, opponent, n_games)
        # print(rewards)

        if plot:
            axs[3*i].hist(rewards)
            axs[3*i].set_title(f"Results of playing with {opponent}: Average reward = {np.mean(rewards)}")

            axs[3*i + 1].plot(range(500), rewards[:500])
            axs[3*i + 1].set_title(f"progression of first 500 games with {opponent}")


            axs[3*i + 2].plot(range(n_games-500, n_games), rewards[-500:])
            axs[3*i + 2].set_title(f"progression of last 500 games with {opponent}")

    if plot and n_opp>0:
        plt.savefig("demo1.png")

# Uncomment to plot rewards progression and hists of wins/ties/losses for 
# all players
# plot_plays([kris, mrugesh, abbey, quincy], plot=True)



    # Note 
    # prev 2 opp plays + my last action seems to work best, coupled with 0.5 LR
    # 1. gamma and 1. temp
    # increased states introduce noise, anad i haven't been able to get 
    # stats from any other combo (lost and forgot this one for a while, 
    # hence writing it down)
    # 0.5 LR introduces a bit of noise in the form of oscillating betn tie and win
    # but still better to oscillate betn tie and win instead of converging on some
    # bots but being random for others

    # # Adaptive lR - lowered win rates 
    # if np.sum(rewards)/len(rewards) > 0.2:
    #     LR = LR + 0.1 if LR < 0.7 else LR
    #     # GAMMA = GAMMA - 0.01 if GAMMA > 0.9 else GAMMA
    # else:
    #     LR = LR - 0.1 if LR > 0.1 else LR

    # # Adaptive temperature - didn't work either 
    # if np.sum(rewards) / len(rewards) < 0.2:
    #     # Strategy not working - introduce more noise - inc temp
    #     temp += 0.1
    # else:
    #     temp = 1.

    # # Epsilon greedy - inconsistent and lower win rate than
    # if not EPSILON_GREEDY:
    #     action = np.random.choice(actions, p=softmax(q_matrix[s_next], temp))
    # else:
    #     if random.random() <= epsilon:
    #         action = random.choice('RPS')
    #     else:
    #         action = actions[np.argmax(q_matrix[state_ids[curr_state]])]

