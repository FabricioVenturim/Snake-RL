import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 80

class SnakeGame:
    def __init__(self, w=640, h=480):
        """Constructor for SnakeGame class.

        Args:
            w (int, optional): width of the game window. Defaults to 640.
            h (int, optional): height of the game window. Defaults to 480.
        """
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        """Resets the game.
        """
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def get_position(self):
        """Returns the position of the snake.
        """
        return self.head.x // BLOCK_SIZE, self.head.y // BLOCK_SIZE

    def _place_food(self):
        """Places the food at a random position.
        """
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def is_collision(self, pt=None):
        """Checks if the snake collides with the wall or itself.

        Args:
            pt (Point, optional): point to check. Defaults to None.
        Returns:
            _type_: True if collision, False otherwise
        """

        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            print("colisão")
            return True
        if pt in self.snake[1:]:
            print("colisão")
            return True
        return False
    
    def is_collision_modify(self, pt=None):
        """Checks if the snake collides with the wall or itself.

        Args:
            pt (_type_, optional): point to check. Defaults to None.

        Returns:
            _type_: True if collision, False otherwise
        """        
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # retira o primiero e ultimo elemento da lista
        if pt in self.snake[1:]:
            return True
        return False

    def play_step(self, action):
        """Plays one step of the game.

        Args:
            action (_type_): action to take
            
        Returns:
            _type_: reward, game_over, score
        """

        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        self._move(action)
        self.snake.insert(0, self.head)
        reward = 0
        game_over = False
        if self.is_collision():
            game_over = True
            reward = -10
            return reward, game_over, self.score
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    def _move(self, action):
        """Moves the snake according to the action.

        Args:
            action: action to take
        """        

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
        self.direction = new_dir

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

    def _update_ui(self):
        """Updates the game window.
        """        

        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        text_moves = font.render("Moves: " + str(self.frame_iteration), True, WHITE)
        self.display.blit(text_moves, [0, 20])
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    # apenas para solução com algoritmo
    # verifica se tal movimento gera colisão
    def is_collision_after_move(self, action, last_head):
        """Checks if the snake collides with the wall or itself after a move.

        Args:
            action (_type_): action to take
            last_head (_type_): last head position
        
        Returns:
            _type_: True if collision, False otherwise

        """

        x_last, y_last = last_head[0], last_head[1]
        x, y = self.head.x, self.head.y
        if x / BLOCK_SIZE < x_last and action == [1,0,0]:
            x -= BLOCK_SIZE
        elif x / BLOCK_SIZE < x_last and action == [0,1,0]:
            y -= BLOCK_SIZE
        elif x / BLOCK_SIZE < x_last and action == [0,0,1]:
            y += BLOCK_SIZE

        elif x / BLOCK_SIZE > x_last and action == [1,0,0]:
            x += BLOCK_SIZE
        elif x / BLOCK_SIZE > x_last and action == [0,1,0]:
            y += BLOCK_SIZE
        elif x / BLOCK_SIZE > x_last and action == [0,0,1]:
            y -= BLOCK_SIZE

        elif y / BLOCK_SIZE < y_last and action == [1,0,0]:
            y -= BLOCK_SIZE
        elif y / BLOCK_SIZE < y_last and action == [0,1,0]:
            x += BLOCK_SIZE
        elif y / BLOCK_SIZE < y_last and action == [0,0,1]:
            x -= BLOCK_SIZE

        elif y / BLOCK_SIZE > y_last and action == [1,0,0]:
            y += BLOCK_SIZE
        elif y / BLOCK_SIZE > y_last and action == [0,1,0]:
            x -= BLOCK_SIZE
        elif y / BLOCK_SIZE > y_last and action == [0,0,1]:
            x += BLOCK_SIZE
        
        point = Point(x, y)
        
        #retorne se tem coliçã e a distancia até fruta:
        # calcula a distancia até a fruta
        if not self.is_collision_modify(point):
            dist_x = abs(self.food.x - point.x)
            dist_y = abs(self.food.y - point.y)
            dist = dist_x + dist_y
        else:
            dist = 1000000
        return self.is_collision_modify(point), dist

        

if __name__ == "__main__":
    game = SnakeGame()
    while True:
        action = [1, 0, 0]  # initially, snake moves right
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        reward, game_over, score = game.play_step(action)
        if game_over:
            game.reset()
        pygame.time.delay(50)