import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

# Initialize pygame
pygame.init()
font = pygame.font.SysFont('arial', 25)

# Define directions for snake movement
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Define a Point named tuple for x and y coordinates
Point = namedtuple('Point', 'x, y')

# Define some colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
GREEN1 = (0, 255, 100)
GREEN2 = (0, 155, 0)
BLACK = (0, 0, 0)

# Constants for block size and speed
BLOCK_SIZE = 20
SPEED = 500

# Main game class
class SnakeGame:
    # Initialize the game
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    # Reset game to its initial state
    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    # Place the food in a random location on the board
    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    # Check for collision with walls or itself
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Collision with walls
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # Collision with itself
        if pt in self.snake[1:]:
            return True
        return False

    # Play one step of the game
    def play_step(self, action):
        self.frame_iteration += 1
        # Handle quitting the game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        # Move the snake based on the action
        self._move(action)
        self.snake.insert(0, self.head)
        reward = 0
        game_over = False
        # Check for collisions or a long game without progress
        if self.is_collision():
            print("Collision")
            game_over = True
            reward = -10
            return reward, game_over, self.score
        if self.frame_iteration > 100 * len(self.snake) * (self.score + 1):
            print("No progress")
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # Check if the snake ate the food
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
            self.snake.pop() 
        else:
            self.snake.pop() 
        # Update the UI
        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    # Move the snake based on the provided action
    def _move(self, action):
        # List that maps the four possible directions in clockwise order
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        # Captures the index of the snake's current direction
        idx = clock_wise.index(self.direction)
        
        # action is a list of 3 values. Each combination of values represents a specific action
        # [1, 0, 0] means continue in the same direction
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        # [0, 1, 0] means turn clockwise
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        # Any other value means turn counter-clockwise
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
        self.direction = new_dir

        # Update the head position based on the direction
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

    # Update the game display/UI
    def _update_ui(self):
        # Fill background with black
        self.display.fill(BLACK)
        # Draw the snake
        for pt in self.snake:
            if pt == self.head:
                pygame.draw.rect(self.display, GREEN2, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, GREEN1, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
            else:
                pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
        # Draw the food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        # Display the score
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

# Run the game if this script is executed
if __name__ == "__main__":
    game = SnakeGame()
    while True:
        action = [1, 0, 0]  # initially, snake moves right
        # Handle quitting the game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        reward, game_over, score = game.play_step(action)
        # Reset game if it's over
        if game_over:
            game.reset()
        pygame.time.delay(50)

