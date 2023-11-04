# import torch #pytorch
import os
import sys
FATHER_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(FATHER_DIRECTORY)

import numpy as np
from game.snake import SnakeGame, Direction, Point 
from RL.RL import QLearing
import math

class Agent:
    def __init__(self, type_state, ng = 0, epsilon = 0.05, alpha = 0.8, gamma = 0.9):
        self.n_games = ng  # Counter for the number of games played
        self.type_state = type_state
        var = (type_state == "STATE4") or (type_state == "STATE5") 
        self.qlearning = QLearing([(1,0,0),(0,1,0),(0,0,1)], epsilon = epsilon, alpha = alpha, gamma = gamma, leng = var)

    def get_state(self, game):
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

        # Checking if there's danger in the current direction
        danger_up = (dir_r and game.is_collision(point_r)) or (dir_l and game.is_collision(point_l)
               ) or(dir_u and game.is_collision(point_u)) or (dir_d and game.is_collision(point_d))

        # Checking danger to the right
        danger_right = (dir_u and game.is_collision(point_r)) or(dir_d and game.is_collision(point_l)
                  ) or (dir_l and game.is_collision(point_u)) or (dir_r and game.is_collision(point_d))

        # Checking danger to the left
        danger_left = (dir_d and game.is_collision(point_r)) or (dir_u and game.is_collision(point_l)
                 ) or (dir_r and game.is_collision(point_u)) or(dir_l and game.is_collision(point_d))
        
        if self.type_state == "STATE1":
            # Constructing the state from various conditions
            state = [
                danger_up, danger_right, danger_left, # danger 

                dir_l, dir_r, dir_u, dir_d,  # Current direction of the snake

                # Food's relative position
                game.food.x < game.head.x,  
                game.food.x > game.head.x,  
                game.food.y < game.head.y,  
                game.food.y > game.head.y   
            ]

        if self.type_state == 'STATE2':
            # quadrant is the difference of the position between target and snake head
            quadrant = [game.food.x - game.head.x, game.food.y - game.head.y]

            if dir_u:
                pass
            elif dir_r:
                temp = quadrant[0]
                quadrant[0] = quadrant[1]
                quadrant[1] = -temp
            elif dir_d:
                quadrant[0] = -quadrant[0]
                quadrant[1] = -quadrant[1]
            elif dir_l:
                temp = quadrant[0]
                quadrant[0] = -quadrant[1]
                quadrant[1] = temp
            
            # Constructing the state from various conditions
            state = [danger_up,danger_right,danger_left,tuple(quadrant)]

        if self.type_state == 'STATE3':

            # quadrant is the difference of the position between target and snake head
            quadrant = [game.food.x - game.head.x, game.food.y - game.head.y]

            if dir_u:
                quadrant[0] = self.get_direction(quadrant[0])
                quadrant[1] = self.get_direction(quadrant[1])
            elif dir_r:
                temp = quadrant[0]
                quadrant[0] = self.get_direction(quadrant[1])
                quadrant[1] = self.get_direction(-temp)
            elif dir_d:
                quadrant[0] = self.get_direction(-quadrant[0])
                quadrant[1] = self.get_direction(-quadrant[1])
            elif dir_l:
                temp = quadrant[0]
                quadrant[0] = self.get_direction(-quadrant[1])
                quadrant[1] = self.get_direction(temp)
            
            # Constructing the state from various conditions
            state = [danger_up, danger_right, danger_left , tuple(quadrant)]

        if self.type_state == "STATE4":
            # quadrant is the difference of the position between target and snake head
            quadrant = [game.food.x - game.head.x, game.food.y - game.head.y]

            if dir_u:
                pass
            elif dir_r:
                temp = quadrant[0]
                quadrant[0] = quadrant[1]
                quadrant[1] = -temp
            elif dir_d:
                quadrant[0] = -quadrant[0]
                quadrant[1] = -quadrant[1]
            elif dir_l:
                temp = quadrant[0]
                quadrant[0] = -quadrant[1]
                quadrant[1] = temp
            
            # Constructing the state from various conditions
            state = [danger_up,danger_right,danger_left,tuple(quadrant),math.floor(game.score/5)]

        if self.type_state == 'STATE5':

            # quadrant is the difference of the position between target and snake head
            quadrant = [game.food.x - game.head.x, game.food.y - game.head.y]

            if dir_u:
                quadrant[0] = self.get_direction(quadrant[0])
                quadrant[1] = self.get_direction(quadrant[1])
            elif dir_r:
                temp = quadrant[0]
                quadrant[0] = self.get_direction(quadrant[1])
                quadrant[1] = self.get_direction(-temp)
            elif dir_d:
                quadrant[0] = self.get_direction(-quadrant[0])
                quadrant[1] = self.get_direction(-quadrant[1])
            elif dir_l:
                temp = quadrant[0]
                quadrant[0] = self.get_direction(-quadrant[1])
                quadrant[1] = self.get_direction(temp)
            
            # Constructing the state from various conditions
            state = [danger_up, danger_right, danger_left , tuple(quadrant),math.floor(game.score/5)]

        if self.type_state=='STATETAB':
            state = [game.head.x,game.head.y]
        
        return state  # Convert the state to an array and return

    
    def get_action(self, state):
        return self.qlearning.get_A(state) # Return the chosen action
    
    def update_Q(self, state, action, new_state, reward):
        self.qlearning.update_Q(state, action, new_state, reward)
    
    def get_direction(self,value):
        if value == 0:
            return 0
        if value > 0:
            return 1
        if value < 0:
            return -1
    