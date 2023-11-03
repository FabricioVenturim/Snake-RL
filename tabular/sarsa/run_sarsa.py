import os
import sys

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
GRANDFATHER_DIRECTORY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(GRANDFATHER_DIRECTORY)

from agent_sarsa import Agent
from game.snake import SnakeGame, WeightRewards
from helper.plot import plot
from ast import literal_eval

LIST_TYPES = ['STATE1','STATE2','STATE3','STATE4']
TYPE_STATE = LIST_TYPES[0]
SCORES_FILE_NAME = CURRENT_DIRECTORY + '\\results\\'+TYPE_STATE+'_scores.txt'
QTABLE_FILE_NAME = CURRENT_DIRECTORY + '\\results\\'+TYPE_STATE+'_Q.txt'

N_GAMES_TRAIN = 50000

STARTING_0 = True

def define_weights():
    w = WeightRewards()
    w.default = -1
    w.died_wall = -10
    w.died_time = -10
    w.died_ifself = -10
    w.ate = 30
    return w

# Define the training function
def train():
    # Lists to store scores and mean scores for plotting

    if STARTING_0:
        l = []
    else:
        arquivo = open(SCORES_FILE_NAME, 'r')
        txt = arquivo.read()
        arquivo.close()
        l = list(literal_eval(txt))

    plot_scores = l # []
    plot_mean_scores = []
    total_score = sum(plot_scores) # 0
    record = len(l)>0 and max(l[0]) or 0  # 0 # Record score

    # Initialize the agent and the Snake game
    agent = Agent(ng=len(plot_scores), type_state = TYPE_STATE)
    game = SnakeGame(w=400, h=400, rw=define_weights())

    if not STARTING_0:
        agent.sarsa.load_Q(QTABLE_FILE_NAME)

    # Get the current state of the game
    state_old = agent.get_state(game)

    # Decide on an action based on the current state
    action_old = agent.get_action(tuple(state_old))

    # Training loop
    while True:

        # Play a step in the game with the chosen action
        reward, done, score = game.play_step(action_old)

        # Get the new state after taking the action
        state_new = agent.get_state(game)

        action_new = agent.get_action(tuple(state_new))

        agent.update_Q(tuple(state_old), action_old, 
                      tuple(state_new), action_new, reward)
        
        state_old = state_new
        action_old = action_new

        # If the game is over
        if done:
            # Reset the game for a new episode
            game.reset()

            # Increment the number of games played
            agent.n_games += 1

            # Update the record score if the current score exceeds it
            if score > record:
                record = score
        
            # Update scores list and calculate mean score for plotting
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

            # Display game statistics
            print(TYPE_STATE, 'Game:', agent.n_games, 'Score:', score, 'Record:', record, "Mean:",round(mean_score,3))

            # Save the model with the new record
            if agent.n_games % 50 == 0:
                agent.sarsa.save_Q(QTABLE_FILE_NAME)
                with open(SCORES_FILE_NAME, 'w') as f:
                    f.write(str(plot_scores))
                print('salvo')

            # Plot the scores and mean scores
            # plot(plot_scores, plot_mean_scores)

            if agent.n_games == N_GAMES_TRAIN:
                break

if __name__ == '__main__':
    train()
    TYPE_STATE = LIST_TYPES[1]
    train()
    TYPE_STATE = LIST_TYPES[2]
    train()
    TYPE_STATE = LIST_TYPES [3]
    train()

