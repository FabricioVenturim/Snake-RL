import os
import sys

snake_rl_path = os.path.abspath(os.path.join(os.path.dirname("run_model3")))
sys.path.append(snake_rl_path)

import torch #pytorch
import random
import numpy as np
from collections import deque #data structure to store memory
from game.snake_without_growing import SnakeGame, Direction, Point 
from model.model import Linear_QNet, QTrainer 
from helper.plot import plot 


# Constants
MAX_MEMORY = 100_000  # Maximum memory capacity for agent's experience storage
BATCH_SIZE = 10000  # Number of experiences used for training in each batch
LR = 0.001  # Learning rate for the neural network training

class Agent:
    def __init__(self):
        self.n_games = 0  # Counter for the number of games played
        self.epsilon = 0  # Exploration rate for making random moves
        self.gamma = 0.9  # Discount rate for considering future rewards
        self.memory = deque(maxlen=MAX_MEMORY)  # Storage for the agent's experiences
        self.model = Linear_QNet(11, 256, 3)  # Neural network model instantiation
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)  # Trainer object for model training

    def get_state(self, game):
        """ Gets the current state of the game.
        """

        # Extracting the snake's head position and nearby points
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        # Current direction of the snake
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Constructing the state from various conditions
        state = [
            # Checking if there's danger in the current direction
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Checking danger to the right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Checking danger to the left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            dir_l, dir_r, dir_u, dir_d,  # Current direction of the snake

            # Food's relative position
            game.food.x < game.head.x,  
            game.food.x > game.head.x,  
            game.food.y < game.head.y,  
            game.food.y > game.head.y   
        ]
        return np.array(state, dtype=int)  # Convert the state to an array and return

    def remember(self, state, action, reward, next_state, done):
        """ Stores the agent's experiences in memory.
        
        args:
            state: current state of the game
            action: action taken by the agent
            reward: reward received by the agent
            next_state: next state of the game
            done: whether the game is over or not
        """

        # Store the experience in memory
        self.memory.append((state, action, reward, next_state, done)) 

    def train_long_memory(self):
        """ Trains the model on a batch of experiences from memory.
        """

        # Train on a batch from the stored experiences
        mini_sample = random.sample(self.memory, BATCH_SIZE) if len(self.memory) > BATCH_SIZE else self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        # Train the model immediately after the agent takes an action
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Decide the next action based on epsilon-greedy policy
        self.epsilon = 80 - self.n_games  # Adjusting exploration rate based on games played
        final_move = [0, 0, 0]
        
        # Choose a random action with epsilon probability
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Otherwise, choose the action with the highest Q-value prediction
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move  # Return the chosen action
    
    